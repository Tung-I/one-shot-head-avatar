import os
import cv2
import yaml
import torch
import argparse
import importlib
import logging
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer


from utils.cfgnode import CfgNode
from utils.camera_utils import sample_random_camera
from utils.utils import torch_to_np


TRI_DIR = '../triplanes'
RENDER_DIR = '../renders'

class TriplaneRenderer():
    """
    Render tri-planes to RGB video frames
    """
    def __init__(self, configargs):
        with open(configargs.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)
        cfg.device = f'cuda:0'
        self.cfg = cfg
        self.model = None
        
        self.tri_dir = TRI_DIR
        self.frame_limit = configargs.frame_limit
        self.render_dir = RENDER_DIR
        self.checkpoint = configargs.checkpoint
        self.device = cfg.device

    def load_model(self, save_video=True):
        checkpoint_path = self.checkpoint
        assert os.path.exists(checkpoint_path)

        # Load pre-trained weights
        checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
        self.iter_num = checkpoint['iter']
        saved_state_dict = checkpoint['model_state_dict']

        # Client-side pretrained weights: "neural_layers" and "decoder"
        print('===> Loading client-side pretrained weights')
        client_model = importlib.import_module('models.model_client').ClientModel(self.cfg, test=True)
        client_state_dict = {}
        client_state_dict = {
            k:v for k, v in saved_state_dict.items() if (k.startswith('neural_layers') or k.startswith('decoder'))
        }
        client_model.load_state_dict(client_state_dict)
        self.model= client_model.to(self.device)
        self.model.eval()

        print("===> Loaded checkpoint from {} at iteration {}.".format(checkpoint_path, self.iter_num))
        return

    def test(self, save_video=True):
        device = self.device

        if save_video:
            out_path = os.path.join(self.render_dir, f'rendering.mp4')
            fps = 15
            n_col = 2
            cap_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512*n_col, 512))

        for i in tqdm(range(self.frame_limit)):
       
            data = {}
            plane = np.load(os.path.join(self.tri_dir, '{0:04d}_tri.npy'.format(i)))
            data['triplane'] = torch.from_numpy(plane).to(device).to(torch.float32)

            # Sample random views to render
            batch_size = 1
            c = sample_random_camera(i, device)
            data['pose_target'] = c.repeat(batch_size, 1)

            # Render tri-plane
            # start = timer()
            with torch.no_grad():
                outputs = self.model(data)
            # print("Time taken to render a frame: {}s".format(timer() - start))

            pred_novel_view = torch_to_np(outputs['high_res'])

            if save_video:
                drive_frame = cv2.imread(os.path.join(self.tri_dir, '{0:04d}_drive.png'.format(i)))
                out_frame = np.concatenate([drive_frame, pred_novel_view], axis=1)
                out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                cap_out.write(out_frame)

        cap_out.release()
        print("===> Rendering done.")


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

    # Create directories to save output
    os.makedirs(RENDER_DIR, exist_ok=True)
    renderer = TriplaneRenderer(configargs)
    renderer.load_model()
    renderer.test()
    


if __name__ == "__main__":
    main()

