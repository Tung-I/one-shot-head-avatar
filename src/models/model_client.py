import torch
from models.volumetric_rendering.renderer import ImportanceRenderer
from models.volumetric_rendering.ray_sampler import RaySampler
from models.networks import OSGDecoder
from timeit import default_timer as timer

import numpy as np
from PIL import Image
from GFPGAN.GFPUpsampler import GFPUpsampler

class ClientModel(torch.nn.Module):
    def __init__(self, cfg, test=False):
        super(ClientModel, self).__init__()
        self.cfg = cfg

        # decoder
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32})

        # renderer
        self.ray_sampler = RaySampler()
        self.renderer = ImportanceRenderer()
        self.render_resolution = cfg.models.render_resolution
        self.rendering_kwargs = {'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 'gpc_reg_prob': 0.5, 'c_scale': 1.0, 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True, 'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2]}

        # SR
        self.neural_layers = GFPUpsampler(cfg.experiment.GFP_path)
        if hasattr(cfg.experiment, 'freeze_gfp') and cfg.experiment.freeze_gfp == 1:
            for param in self.neural_layers.parameters():
                param.requires_grad = False

    def render_plane(self, plane, ray_origins, ray_directions, render_resolution):
        """
        Render a tri-plane at given camera view.
        - Inputs:
          - plane: (b, 3, 32, 256, 256)
          - ray_origins: (b, h*w, 3)
          - ray_directions: (b, h*w, 3)
        - Outputs:
          - rgb_low_res: (b, 3, self.render_resolution ,self.render_resolution)
          - depth_image: (b, 1, self.render_resolution ,self.render_resolution)
        """
        feature_samples, depth_samples, _ = self.renderer(plane, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        depth_image = depth_samples.permute(0, 2, 1).reshape(-1, 1, render_resolution, render_resolution)
        color_feat = feature_samples.view(-1, render_resolution, render_resolution, 32).permute(0, 3, 1, 2)
        rgb_low_res = color_feat[:, :3]
        return rgb_low_res, depth_image

    def forward(self, data):
        outputs = {}
        plane = data['triplane']

        pose_tar = data['pose_target']
        tar_cam2world = pose_tar[:, :16].view(-1, 4, 4)
        tar_intrinsics = pose_tar[:, 16:25].view(-1, 3, 3)
        tar_ray_origins, tar_ray_directions = self.ray_sampler(tar_cam2world, tar_intrinsics, self.render_resolution)

        # Render the tri-plane (26 ms)
        start = timer()
        rgb_low_res, depth_image = self.render_plane(plane, tar_ray_origins, tar_ray_directions, self.render_resolution)
        print("Time taken to render a frame: {}s".format(timer() - start))

        # SR
        start = timer()
        neural_rendering = self.neural_layers(rgb_low_res)[0]
        print("Time taken to upsample a frame: {}s".format(timer() - start))

        outputs['high_res'] = neural_rendering
        return outputs
