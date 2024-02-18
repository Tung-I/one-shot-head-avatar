import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import PIL.Image
import json

def torch_to_np(img):
    """
    Convert a torch tensor image to a numpy image.
        - Input: torch.Tensor of size (1, c, h, w) in range [-1, 1].
        - Output: numpy array of size (h, w, c) in range [0, 255].
    """
    img = (img + 1) / 2.0
    img = img * 255
    img = img.clamp_(0, 255)
    img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = np.array(img, dtype = np.uint8)
    return img

def load_raw_image(fname):
    with open(fname, 'rb') as f:
        image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    image = image.transpose(2, 0, 1) # HWC => CHW
    return image

def load_parsing_image(parsing_path):
    mask = PIL.Image.open(parsing_path)
    mask = np.asarray(mask) / 255.0
    return mask
