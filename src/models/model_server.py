# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from timeit import default_timer as timer

from transformers import SegformerForSemanticSegmentation
from models.networks import Embed2Plane
from deep3d_modules.facerecon_model import FaceReconModel
from deep3d_modules.render import MeshRenderer

import torch.onnx



class ServerModel(torch.nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.device = 'cuda:0'

        # Expression branch 
        # self.expr_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.expr_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", torchscript=True)
        self.expr_net.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.expr_net.decode_head.batch_norm = nn.Identity()
        self.expr_net.decode_head.activation = nn.Identity()
        self.expr_net.decode_head.dropout = nn.Identity()
        self.expr_net.decode_head.classifier = nn.Identity()
        self.embedding2expr = Embed2Plane(256, 96)

        # 3DMM model
        self.deep3D = FaceReconModel()
        self.deep3D.setup()
        self.deep3D.device = self.device
        self.deep3D.eval()
        self.deep3D.facemodel.to(self.device)
        self.deep3D.net_recon.to(self.device)
        focal = 1015.0
        center = 112.0
        z_near = 5.0
        z_far = 15.0
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        self.renderer = MeshRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center))
        for param in self.deep3D.net_recon.parameters():
            param.requires_grad = False

        # Registered buffer: param that should be be saved and restored in the state_dict, but not trained by the optimizer
        cano_pose = torch.tensor([[ 0.9997,  0.0059,  0.0250, -0.0731,  0.0081, -0.9961, -0.0882,  0.2462,
          0.0244,  0.0884, -0.9958,  2.6878,  0.0000,  0.0000,  0.0000,  1.0000,
          4.2647,  0.0000,  0.5000,  0.0000,  4.2647,  0.5000,  0.0000,  0.0000,
          1.0000]])
        
        img = np.array(Image.open('cache/cano_exemplar.png')).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        
        u = torch.linspace(-1, 1, steps = 128)
        v = torch.linspace(-1, 1, steps = 128)
        u, v = torch.meshgrid(u, v, indexing='xy')
        uv = torch.stack((u, v), dim = -1).view(-1, 2)

        self.register_buffer("cano_pose", cano_pose) # 1, 25
        self.register_buffer("cano_exemplar", (img / 255.0) * 2 - 1)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([[0.229, 0.224, 0.225]]).view(1, 3, 1, 1))
        self.register_buffer('uv', uv.unsqueeze(0))
        self.cano_render_resolution = 128
    
    def load_model(self, state_dict):
        def key_filter(key):
            if key.startswith('expr_net') or key.startswith('embedding2expr') or \
                key.startswith('deep3D') or key.startswith('renderer') or \
                key.startswith('exp_loss_fn') or key.startswith('cano_pose') or \
                key.startswith('cano_exemplar') or key.startswith('image_mean') or \
                key.startswith('image_std') or key.startswith('uv'):
                return True
            else:
                return False
        # def key_filter(key):
        #     if key.startswith('deep3D') or key.startswith('renderer') or \
        #         key.startswith('exp_loss_fn') or key.startswith('cano_pose') or \
        #         key.startswith('cano_exemplar') or key.startswith('image_mean') or \
        #         key.startswith('image_std') or key.startswith('uv'):
        #         return True
        #     else:
        #         return False

        _state_dict = {
            k:v for k, v in state_dict.items() if key_filter(k)
        }
        self.load_state_dict(_state_dict)

        # Caching
        cano_plane = np.load('cache/cano.npy')
        app_plane = np.load('cache/app.npy')
        cano_plane_cached = torch.tensor(cano_plane).to(self.device)
        app_plane_cached = torch.tensor(app_plane).to(self.device)
        img_src = np.load('cache/img_src.npy')
        img_src = torch.from_numpy(img_src[None, ...]).to(self.device).to(torch.float32) / 127.5 - 1
        self.register_buffer("cano_plane_cached", cano_plane_cached)
        self.register_buffer("app_plane_cached", app_plane_cached)
        self.register_buffer("img_src", img_src)

    def preprocess(self, img):
            return F.interpolate((img + 1) / 2.0, size = (224, 224), mode = 'bilinear', align_corners = False)
    
    def forward(self, img_tar):

        id_img = self.preprocess(self.img_src)
        pose_img = self.preprocess(self.cano_exemplar)
        tar_img = self.preprocess(img_tar)
        img = torch.cat((id_img, pose_img, tar_img))
        with torch.no_grad():
            output_coeff = self.deep3D.net_recon(img)
        output_coeff[0, 80: 144] = output_coeff[2, 80:144]  # replace expression with target expression
        output_coeff[0, 224: 227] = output_coeff[1, 224: 227]  # replace head pose with target head pose
        output_coeff[0, 254:] = output_coeff[1, 254:]

        with torch.no_grad():
            pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff[0:1])
            pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
        
        sudo = pred_face * 2 - 1
        sudo = F.interpolate(sudo, size=self.img_src.shape[-2:], mode='bilinear', align_corners=False)
        sudo = ((sudo + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        # return sudo
        
        # inputs = {}
        # inputs['pixel_values'] = sudo
        # inputs['output_hidden_states'] = True
        # expr_feat = self.expr_net(**inputs).logits

        expr_feat = self.expr_net(pixel_values=sudo, output_hidden_states=True, return_dict=True).logits
        expr_planes = self.embedding2expr(expr_feat)

        return self.cano_plane_cached + expr_planes + self.app_plane_cached

        
        
    
    # def sudo_forward(self, data):
    #     img_tar = data['img_target']
    #     img_src = data['img_src']

    #     sudo = self.get_sudo(img_src, self.cano_exemplar, img_tar)
    #     sudo = F.interpolate(sudo, size=img_src.shape[-2:], mode='bilinear', align_corners=False)
    #     return sudo

    # def get_sudo(self, id_img, pose_img, tar_img, exp_transfer=True):
    #     """
    #     Given an image and a desired expression coefficient, return sudo 3DMM rendering
    #     """
    #     def preprocess(img):
    #         return F.interpolate((img + 1) / 2.0, size = (224, 224), mode = 'bilinear', align_corners = False)
    #     b, c, h, w = id_img.shape
    #     id_img = preprocess(id_img)
    #     pose_img = preprocess(pose_img)
    #     tar_img = preprocess(tar_img)
    #     img = torch.cat((id_img, pose_img, tar_img))
    #     with torch.no_grad():
    #         output_coeff = self.deep3D.net_recon(img)

    #     if exp_transfer:
    #         # replace expression with target expression
    #         output_coeff[0, 80: 144] = output_coeff[2, 80:144]
    #     # replace head pose with target head pose
    #     output_coeff[0, 224: 227] = output_coeff[1, 224: 227]
    #     output_coeff[0, 254:] = output_coeff[1, 254:]

    #     with torch.no_grad():
    #         pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff[0:1])
    #         pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
    #     return pred_face * 2 - 1

    # def export_exp_encoder(self, data):
    #     img_tar = data['img_target']
    #     img_src = data['img_src']

    #     sudo = self.get_sudo(img_src, self.cano_exemplar, img_tar)
    #     sudo = F.interpolate(sudo, size=img_src.shape[-2:], mode='bilinear', align_corners=False)

    #     torch.onnx.export(self.exp_encoder,               # model being run
    #               sudo,                         # model input (or a tuple for multiple inputs)
    #               "../exp_encoder.onnx",   # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=True,  # whether to execute constant folding for optimization
    #               input_names = ['input'],   # the model's input names
    #               output_names = ['output'] # the model's output names
    #               )