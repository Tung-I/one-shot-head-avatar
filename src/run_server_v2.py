import os
import cv2
import yaml
import torch
import argparse
import importlib
import logging
import numpy as np
from tqdm import tqdm
import PIL.Image
import json
import natsort
from timeit import default_timer as timer

from utils.cfgnode import CfgNode
from utils.utils import torch_to_np, load_raw_image, load_parsing_image

SRC_DIR = '/home/ubuntu/datasets/cap/src'
SRC_FNAME = os.path.join(SRC_DIR, 'source.png')
SRC_MASK_FNAME = os.path.join(SRC_DIR, 'source_matted.png')
SRC_POSE_FNAME = os.path.join(SRC_DIR, 'camera.json')

VIDEO_DIR = '/home/ubuntu/datasets/cap/images'
MATTING_DIR = '/home/ubuntu/datasets/cap/matting'

TRI_DIR = '../triplanes'


class VideoFrameCollector():
    def __init__(self, frame_limit=60):

        self.tgt_fnames = []
        self.tgt_mask_fnames = []
        self.frame_limit = frame_limit
        self.video_dir = VIDEO_DIR
        self.matting_dir = MATTING_DIR

        self.collect_video_frames()

    def collect_video_frames(self):
        for i in range(self.frame_limit):
            # Images
            frame_fname = os.path.join(self.video_dir, 'frame_{0:04d}.png'.format(i))
            self.tgt_fnames.append(frame_fname)
            # Masks
            mask_fname = os.path.join(self.matting_dir, 'frame_{0:04d}_matte.png'.format(i))
            self.tgt_mask_fnames.append(mask_fname)

        print("Found {} drive video frames in total.".format(len(self.tgt_fnames)))

        
class OneShotHeadAvatar():
    def __init__(self, configargs):
        with open(configargs.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)
        cfg.device = f'cuda:0'
        self.cfg = cfg
        self.frame_limit = configargs.frame_limit
        self.checkpoint = configargs.checkpoint
        self.device = cfg.device
        self.tri_dir = TRI_DIR

        self.src_fname = SRC_FNAME
        self.src_mask_fname = SRC_MASK_FNAME
        self.src_pose_fname = SRC_POSE_FNAME

        # Load source image and mask
        src_img = load_raw_image(self.src_fname)  # (3, 512, 512)
        src_mask = load_parsing_image(self.src_mask_fname)  # (512, 512)
        assert src_img.dtype == np.uint8
        src_img_masked = src_img * src_mask.reshape(1, src_mask.shape[0], src_mask.shape[1])  # (3, 512, 512)
        self.src_img_masked = torch.from_numpy(src_img_masked[None, ...]).to(self.device).to(torch.float32) / 127.5 - 1
        
        # Load source pose
        with open(self.src_pose_fname, 'rb') as f:
            labels = json.load(f)['labels']
        label_dict = {}
        for label in labels:
            label_dict[label[0]] = [label[1]]
        (img_name, [pose]) = list(label_dict.items())[0]
        src_pose = np.array(pose)
        self.src_pose = torch.from_numpy(src_pose[None, ...]).to(self.device).float()

    def load_model(self):
        checkpoint_path = self.checkpoint
        assert os.path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
        self.iter_num = checkpoint['iter']
        saved_state_dict = checkpoint['model_state_dict']

        # Server-side pretrained weights: load all except 'neural_layers'
        server_model = importlib.import_module('models.model_server').ServerModel(self.cfg)
        server_state_dict = {
            k:v for k, v in saved_state_dict.items() if (not k.startswith('neural_layers') and not k.startswith('expr_upsampler'))
        }
        server_model.load_state_dict(server_state_dict)
        self.model = server_model.to(self.device)
        self.model.eval()

        print("===> Loaded checkpoint from {} at iteration {}.".format(checkpoint_path, self.iter_num))
        return
    
    def lift2tri(self, iteration, data, save_tri=False):
        """
        Lift images to tri-plane
        data: dictionary containing the following
            - img_target: torch.Tensor of size (1, 3, h, w) in range [-1, 1].
            - pose_target: torch.Tensor of size (1, 6) in range [-1, 1].
            - img_src: torch.Tensor of size (1, 3, h, w) in range [-1, 1].
        """
        start = timer()
        with torch.no_grad():
            outputs = self.model(data)  # 38 ms if reuse app-plane
        end = timer()
        print("Time taken to lift a frame: {}".format(end - start))  
    
        # Save tri-plane and drive frame
        if save_tri:
            tri_plane = outputs['triplane'].detach().cpu().numpy()
            np.save(os.path.join(self.tri_dir, '{0:04d}_tri.npy'.format(iteration)), tri_plane)
            drive_frame = torch_to_np(data['img_target'])
            cv2.imwrite(os.path.join(self.tri_dir, '{0:04d}_drive.png'.format(iteration)), drive_frame)
    
    def test(self, tgt_fnames, tgt_mask_fnames):
        device = self.device
        data = {}
        data['pose_src'] = self.src_pose
        data['img_src'] = self.src_img_masked

        for i in tqdm(range(self.frame_limit)):
            tgt_frame = load_raw_image(tgt_fnames[i])
            tgt_mask = load_parsing_image(tgt_mask_fnames[i])
            tgt_img_masked = tgt_frame * tgt_mask.reshape(1, tgt_mask.shape[0], tgt_mask.shape[1])
            tgt_img_masked = torch.from_numpy(tgt_img_masked[None, ...]).to(device).to(torch.float32) / 127.5 - 1
            data['img_target'] = tgt_img_masked

            if i == 0: 
                print("Warmup ...")
                with torch.no_grad():
                    self.model.warmup(data)
            
            self.lift2tri(i, data, save_tri=True)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--frame_limit", type=int, default=60, help="frame number limitation"
    )
    configargs = parser.parse_args()

    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Create directories to save output
    os.makedirs(TRI_DIR, exist_ok=True)

    vfc = VideoFrameCollector(frame_limit=configargs.frame_limit)
    tgt_fnames = vfc.tgt_fnames
    tgt_mask_fnames = vfc.tgt_mask_fnames

    osha = OneShotHeadAvatar(configargs)
    osha.load_model()
    osha.test(tgt_fnames, tgt_mask_fnames)
        


if __name__ == "__main__":
    main()
