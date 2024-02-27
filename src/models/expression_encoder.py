import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from timeit import default_timer as timer
from models.networks import Embed2Plane



class ExpressionEncoder(nn.Module):
    def __init__(self):
        super(ExpressionEncoder, self).__init__()
        self.device = 'cuda:0'

        # Expression branch
        self.expr_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.expr_net.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.expr_net.decode_head.batch_norm = nn.Identity()
        self.expr_net.decode_head.activation = nn.Identity()
        self.expr_net.decode_head.dropout = nn.Identity()
        self.expr_net.decode_head.classifier = nn.Identity()
        self.embedding2expr = Embed2Plane(256, 96)

        # # Registered buffer: param that should be be saved and restored in the state_dict, but not trained by the optimizer
        # cano_pose = torch.tensor([[ 0.9997,  0.0059,  0.0250, -0.0731,  0.0081, -0.9961, -0.0882,  0.2462,
        #   0.0244,  0.0884, -0.9958,  2.6878,  0.0000,  0.0000,  0.0000,  1.0000,
        #   4.2647,  0.0000,  0.5000,  0.0000,  4.2647,  0.5000,  0.0000,  0.0000,
        #   1.0000]])
        # self.register_buffer("cano_pose", cano_pose) # 1, 25
        # img = np.array(Image.open('cache/cano_exemplar.png')).transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # self.register_buffer("cano_exemplar", (img / 255.0) * 2 - 1)
        # self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('image_std', torch.tensor([[0.229, 0.224, 0.225]]).view(1, 3, 1, 1))
        # u = torch.linspace(-1, 1, steps = 128)
        # v = torch.linspace(-1, 1, steps = 128)
        # u, v = torch.meshgrid(u, v, indexing='xy')
        # uv = torch.stack((u, v), dim = -1).view(-1, 2)
        # self.register_buffer('uv', uv.unsqueeze(0))
        # self.cano_render_resolution = 128


    def load_model(self, state_dict):
        # def key_filter(key):
        #     if key.startswith('expr_net') or key.startswith('embedding2expr') or \
        #         key.startswith('exp_loss_fn') or key.startswith('cano_pose') or \
        #         key.startswith('cano_exemplar') or key.startswith('image_mean') or \
        #         key.startswith('image_std') or key.startswith('uv'):
        #         return True
        #     else:
        #         return False
        def key_filter(key):
            if key.startswith('expr_net') or key.startswith('embedding2expr'):
                return True
            else:
                return False

        _state_dict = {
            k:v for k, v in state_dict.items() if key_filter(k)
        }
        self.load_state_dict(_state_dict)

    def forward(self, sudo):
        expr_feat = self.expr_net(pixel_values=sudo, output_hidden_states=True, return_dict=True).logits
        expr_planes = self.embedding2expr(expr_feat)

        return expr_planes

    