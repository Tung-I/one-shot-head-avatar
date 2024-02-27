import os
import numpy as np
from utils.utils import torch_to_np, load_raw_image, load_parsing_image

SRC_DIR = '/home/ubuntu/datasets/cap/src'
SRC_FNAME = os.path.join(SRC_DIR, 'source.png')
SRC_MASK_FNAME = os.path.join(SRC_DIR, 'source_matted.png')
SRC_POSE_FNAME = os.path.join(SRC_DIR, 'camera.json')

IMG_SRC_SAVE_PATH = 'cache/img_src.npy'

def main():
    src_fname = SRC_FNAME
    src_mask_fname = SRC_MASK_FNAME
    img_src_save_path = IMG_SRC_SAVE_PATH

    # Load source image and pose, which are fixed during inference
    src_img = load_raw_image(src_fname)  # (3, 512, 512)
    src_mask = load_parsing_image(src_mask_fname)  # (512, 512)
    assert src_img.dtype == np.uint8
    src_img_masked = src_img * src_mask.reshape(1, src_mask.shape[0], src_mask.shape[1])  # (3, 512, 512)
    # src_img_masked = torch.from_numpy(src_img_masked[None, ...]).to(self.device).to(torch.float32) / 127.5 - 1
    np.save(img_src_save_path, src_img_masked)



if __name__ == "__main__":
    main()