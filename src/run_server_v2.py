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
import onnx
import onnxruntime
from timeit import default_timer as timer

from utils.cfgnode import CfgNode
from utils.utils import torch_to_np, load_raw_image, load_parsing_image

# from models.expression_encoder import ExpressionEncoder

SRC_DIR = '/home/ubuntu/datasets/cap/src'
SRC_FNAME = os.path.join(SRC_DIR, 'source.png')
SRC_MASK_FNAME = os.path.join(SRC_DIR, 'source_matted.png')
SRC_POSE_FNAME = os.path.join(SRC_DIR, 'camera.json')

VIDEO_DIR = '/home/ubuntu/datasets/cap/images'
MATTING_DIR = '/home/ubuntu/datasets/cap/matting'

TRI_DIR = '../triplanes'


class VideoFrameCollector():
    def __init__(self, frame_limit=60):
        """
        Collect frames and masks (*.png) from VIDEO_DIR and MATTING_DIR.
        This is only used for developing and will be replaced by the actual video streams.
        """
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

        
class EncodeTester():
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

        self.load_model()

        # Load source image and pose, which are fixed during inference
        src_img = load_raw_image(self.src_fname)  # (3, 512, 512)
        src_mask = load_parsing_image(self.src_mask_fname)  # (512, 512)
        assert src_img.dtype == np.uint8
        src_img_masked = src_img * src_mask.reshape(1, src_mask.shape[0], src_mask.shape[1])  # (3, 512, 512)
        self.src_img_masked = torch.from_numpy(src_img_masked[None, ...]).to(self.device).to(torch.float32) / 127.5 - 1

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
        self.model = importlib.import_module('models.model_server').ServerModel()

        self.model.load_model(saved_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # self.expr_encoder = ExpressionEncoder()
        # self.expr_encoder.load_model(saved_state_dict)
        # self.expr_encoder.to(self.device)
        # self.expr_encoder.eval()

        print("===> Loaded checkpoint from {} at iteration {}.".format(checkpoint_path, self.iter_num))
        return
    
    def test(self, tar_fnames, tar_mask_fnames, save_tri=True):
        device = self.device

        for i in tqdm(range(self.frame_limit)):
            # Target image masking
            tar_frame = load_raw_image(tar_fnames[i])
            tar_mask = load_parsing_image(tar_mask_fnames[i])
            tar_img_masked = tar_frame * tar_mask.reshape(1, tar_mask.shape[0], tar_mask.shape[1])
            tar_img_masked = torch.from_numpy(tar_img_masked[None, ...]).to(device).to(torch.float32) / 127.5 - 1

            # if trace_output_path:
            #     # Export to TorchScript model
            #     print('Exporting the model to TorchScript ...')
            #     traced_model = torch.jit.trace(self.model, tar_img_masked)
            #     torch.jit.save(traced_model, trace_output_path)
            #     break

            start = timer()
            with torch.no_grad():
                tri_planes = self.model(tar_img_masked)  # 38 ms if reuse app-plane
                # sudo = self.model(tar_img_masked)  # 38 ms if reuse app-plane
                # tri_planes = self.expr_encoder(sudo)
            end = timer()
            print("Time taken to lift a frame: {}".format(end - start))  
        
            # Save tri-plane and drive frame
            if save_tri:
                tri_planes = tri_planes.detach().cpu().numpy()
                np.save(os.path.join(self.tri_dir, '{0:04d}_tri.npy'.format(i)), tri_planes)
                drive_frame = torch_to_np(tar_img_masked)
                cv2.imwrite(os.path.join(self.tri_dir, '{0:04d}_drive.png'.format(i)), drive_frame)

            

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--frame_limit", type=int, help="frame number limitation"
    )
    parser.add_argument(
        "-t", dest="trace_output_path", type=str, default=None, help="Path for saving the TorchScript model."
    )
    configargs = parser.parse_args()

    os.makedirs(TRI_DIR, exist_ok=True)

    vfc = VideoFrameCollector(frame_limit=configargs.frame_limit)
    tgt_fnames = vfc.tgt_fnames
    tgt_mask_fnames = vfc.tgt_mask_fnames

    tester = EncodeTester(configargs)
    tester.test(tgt_fnames, tgt_mask_fnames)

if __name__ == "__main__":
    main()
