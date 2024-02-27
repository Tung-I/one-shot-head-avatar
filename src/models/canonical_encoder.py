import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from kornia import morphology as morph
from transformers import SegformerForSemanticSegmentation
from timeit import default_timer as timer

from models.BiSeNet import BiSeNet
from models.volumetric_rendering.renderer import ImportanceRenderer
from models.volumetric_rendering.ray_sampler import RaySampler, MaskedRaySampler
from models.networks import IDUpsampler, OSGDecoder, Embed2Plane

from losses.threeDMM import ExpLoss


class CanonicalEncoder(nn.Module):
    def __init__(self):
        super(CanonicalEncoder, self).__init__()
        self.template_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.template_net.decode_head.batch_norm = nn.Identity()
        self.template_net.decode_head.activation = nn.Identity()
        self.template_net.decode_head.dropout = nn.Identity()
        self.template_net.decode_head.classifier = nn.Identity()
        self.embedding2neutral = Embed2Plane(256, 96)
        self.freeze_cano = True
        for param in self.template_net.parameters():
            param.requires_grad = False
        for param in self.embedding2neutral.parameters():
            param.requires_grad = False

    def forward(self, img_src, src_mask):

        neutral_sudo, neutral_mask = self.exp_loss_fn.get_neutral_render(img_src)
        neutral_sudo = neutral_sudo * 2 - 1

        im = ((img_src + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        inputs = {}
        inputs['pixel_values'] = im * mask
        inputs['output_hidden_states'] = True
        template_feat = self.template_net(**inputs).logits
        canonical_plane = self.embedding2neutral(template_feat).contiguous()
        if self.freeze_cano:
            canonical_plane = canonical_plane.detach()


