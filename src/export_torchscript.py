import os
import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the TorchScript model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # load the model
    print('Loading the model ...')
    goha = importlib.import_module('models.model_server').ServerModel()
    checkpoint = torch.load(args.ckpt_path, map_location = torch.device('cpu'))
    saved_state_dict = checkpoint['model_state_dict']
    goha.load_model(saved_state_dict)
    goha.eval()

    # export to TorchScript model
    print('Exporting the model to TorchScript ...')
    scripted_model = torch.jit.script(goha)
    torch.jit.save(scripted_model, os.path.join(args.output_path))

