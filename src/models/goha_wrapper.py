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

import onnx
import onnxruntime


def preprocess(img):
    return F.interpolate((img + 1) / 2.0, size = (224, 224), mode = 'bilinear', align_corners = False)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class GOHAWrapper(torch.nn.Module):
    def __init__(self):
        super(GOHAWrapper, self).__init__()
        self.device = 'cuda:0'

        img = np.array(Image.open('cache/cano_exemplar.png')).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        self.register_buffer("cano_exemplar", (img / 255.0) * 2 - 1)
        img_src = np.load('cache/img_src.npy')
        img_src = torch.from_numpy(img_src[None, ...]).to(torch.float32) / 127.5 - 1
        self.register_buffer("img_src", img_src)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer('image_std', torch.tensor([[0.229, 0.224, 0.225]]).view(1, 3, 1, 1).to(self.device))

        print('Loading 3DMM model')
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
        self.renderer.eval()

        print('Create onnxruntime session')
        self.expr_net_session = onnxruntime.InferenceSession("export/expr_net.onnx", providers=['CUDAExecutionProvider'])
        self.embedding2expr_session = onnxruntime.InferenceSession("export/embedding2expr.onnx", providers=['CUDAExecutionProvider'])
        self.deep3D_session = onnxruntime.InferenceSession("export/net_recon.onnx", providers=['CUDAExecutionProvider'])  
    
    def forward(self, img_tar):

        id_img = preprocess(self.img_src)
        pose_img = preprocess(self.cano_exemplar)
        tar_img = preprocess(img_tar)
        img = torch.cat((id_img, pose_img, tar_img))

        onnx_img = {self.deep3D_session.get_inputs()[0].name: to_numpy(img)}
        output_coeff = self.deep3D_session.run(None, onnx_img)[0]
        output_coeff = torch.tensor(output_coeff, device=self.device)

        output_coeff[0, 80: 144] = output_coeff[2, 80:144]  # replace expression with target expression
        output_coeff[0, 224: 227] = output_coeff[1, 224: 227]  # replace head pose with target head pose
        output_coeff[0, 254:] = output_coeff[1, 254:]

        with torch.no_grad():
            pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff[0:1])
            pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, pred_color)
        
        sudo = pred_face * 2 - 1
        sudo = F.interpolate(sudo, size=self.img_src.shape[-2:], mode='bilinear', align_corners=False)
        sudo = ((sudo + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()

        onnx_sudo = {self.expr_net_session.get_inputs()[0].name: to_numpy(sudo)}
        expr_feat = self.expr_net_session.run(None, onnx_sudo)[0]

        onnx_expr_feat = {self.embedding2expr_session.get_inputs()[0].name: expr_feat}
        expr_planes = self.embedding2expr_session.run(None, onnx_expr_feat)[0]

        return expr_planes

      