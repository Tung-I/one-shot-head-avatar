"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from ..util import util 

from. networks import ReconNetWrapper

class PoseEstiModel(BaseModel):
    """
    
    """
    def __init__(self, opt):
        # BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:0') 
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.parallel_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        # self.net_recon = networks.define_net_recon(
        #     net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        # )

        self.net_recon = ReconNetWrapper(opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path)
        self.net_recon = self.net_recon.to(self.device)

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        self.facemodel.to(self.device)  
        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)      
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)


    def compute_losses(self):
        pass
            

    def optimize_parameters(self, isTrain=True):
        pass     

    def compute_visuals(self):
        pass

    def get_coeff(self):
        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        return pred_coeffs