
from PIL import Image
import numpy as np
import os
import cv2
import json

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from timeit import default_timer as timer

from data_preprocessing.Deep3DFaceRecon_pytorch.models import create_model
from data_preprocessing.Deep3DFaceRecon_pytorch.util.preprocess import align_img
from data_preprocessing.Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
from data_preprocessing.Deep3DFaceRecon_pytorch.options import TestOptions

from data_preprocessing.MODNet.src.models.modnet import MODNet

# from mtcnn import MTCNN


SRC_DIR = '/home/ubuntu/datasets/cap/src'
BFM_DIR = '/home/ubuntu/models/BFM'
D3DRECON_CHECKPOINT_DIR = '/home/ubuntu/models/Deep3DFaceRecon/checkpoints'
MODNET_CHECKPOINT_DIR = '/home/ubuntu/models/modnet'
SRC_FNAME = 'source.png'

class PoseEstimator():
    """
    Preprocess the source image for the GOHA model
    - MTCNN face landmark detection
    - Deep3DFaceRecon head pose estimation
    - MODNet background removal

    Outputs:
    - camera.json
    - matted image
    """
    
    def __init__(self):
        self.src_dir = SRC_DIR
        self.device = torch.device(0)
        torch.cuda.set_device(self.device)

        # # MTCNN
        # self.detector = MTCNN()

        # Pose estimation
        self.d3d_opt = TestOptions().parse()  # get d3d test options
        self.d3d_opt.gpu_ids = 0
        self.d3d_opt.name = "face_recon_feat0.2_augment"
        self.d3d_opt.epoch = 20
        self.d3d_opt.bfm_folder = BFM_DIR
        self.d3d_opt.model = 'poseesti'
        self.lm3d_std = load_lm3d(self.d3d_opt.bfm_folder)  # load the template of facial landmarks
        self.d3d_opt.checkpoints_dir = D3DRECON_CHECKPOINT_DIR
        self.d3d_model = create_model(self.d3d_opt)
        self.d3d_model.setup(self.d3d_opt)
        self.d3d_model.device = self.device
        self.d3d_model.eval()
        print('Deep3D reconstruction model loaded successfully')

        # Matting
        matt_ckpt_path = os.path.join(MODNET_CHECKPOINT_DIR, 'modnet_photographic_portrait_matting.ckpt')
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet).cuda()
        self.modnet.load_state_dict(torch.load(matt_ckpt_path))
        self.modnet.eval()

    # def mtcnn(self, fname):
    #     """
    #     MTCNN face detection:
    #      - detect faces and save the landmarks to src_dir/detections
    #     """
    #     image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    #     result = self.detector.detect_faces(image)

    #     if len(result)>0:
    #         index = 0
    #         if len(result)>1: # if multiple faces, take the biggest face
    #             size = -100000
    #             for r in range(len(result)):
    #                 size_ = result[r]["box"][2] + result[r]["box"][3]
    #                 if size < size_:
    #                     size = size_
    #                     index = r

    #         bounding_box = result[index]['box']
    #         keypoints = result[index]['keypoints']
    #         if result[index]["confidence"] > 0.9:

    #             dst = fname.replace('images', 'detections').replace('.png', '.txt')
    #             print(f'Save to MTCNN output to: {dst}')
    #             outLand = open(dst, "w")
    #             outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
    #             outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
    #             outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
    #             outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
    #             outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
    #             outLand.close()

    def estimate_pose(self, im_path, lm_path):
        """
        Parameters:
        - im_path: str, path to the image
        - lm_path: str, path to the landmarks

        Returns:
        - dict, predicted 3DMM coefficients with keys 'angle', 'trans', ..., etc.
        """
        im = Image.open(im_path).convert('RGB')
        W,H = im.size
        lm = np.loadtxt(lm_path).astype(np.float32)
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im, lm, _ = align_img(im, lm, self.lm3d_std)
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)

        data = {
            'imgs': im,
            'lms': lm
        }
        self.d3d_model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            self.d3d_model.forward()           # run inference

        return self.d3d_model.get_coeff()
    
    def modify_pose(self, coeff):
        """
        Parameters:
        - coeff: dict, 3DMM coefficients output by Deep3DRecon

        Returns:
        - dict, modified 3DMM coefficients with keys for GOHA

        """
        angle = coeff['angle']
        trans = coeff['trans'][0]

        R = self.d3d_model.facemodel.compute_rotation(torch.from_numpy(angle).to(self.device))[0]
        R = R.cpu().numpy()

        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R
        c *= 0.27 # normalize camera radius
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 
        pp = 512#112
        w = 1024#224
        h = 1024#224

        count = 0
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0
        K = K.tolist()

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        # Modify the pose and intrinsics for GOHA
        def fix_intrinsics(intrinsics):
            intrinsics = np.array(intrinsics).copy()
            assert intrinsics.shape == (3, 3), intrinsics
            intrinsics[0,0] = 2985.29/700
            intrinsics[1,1] = 2985.29/700
            intrinsics[0,2] = 1/2
            intrinsics[1,2] = 1/2
            assert intrinsics[0,1] == 0
            assert intrinsics[2,2] == 1
            assert intrinsics[1,0] == 0
            assert intrinsics[2,0] == 0
            assert intrinsics[2,1] == 0
            return intrinsics
        
        def fix_pose_orig(pose):
            pose = np.array(pose).copy()
            location = pose[:3, 3]
            radius = np.linalg.norm(location)
            pose[:3, 3] = pose[:3, 3]/radius * 2.7
            return pose

        pose = pose.tolist()
        intrinsics = K
        pose = fix_pose_orig(pose)
        intrinsics = fix_intrinsics(intrinsics)
        label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
        out_dict = {'labels':[]}
        out_dict["labels"].append(['src_img', label])

        return out_dict

    def matting(self, fname):
        """
        Parameters:
        - fname: str, path to the image

        Returns:
        - np.array, matted image
        """
        ref_size = 512
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print(f'Process matting on image: {fname}')
        im = Image.open(fname)

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        _, _, matted = self.modnet(im.cuda() if torch.cuda.is_available() else im, True)

        matted = F.interpolate(matted, size=(im_h, im_w), mode='area')
        matted = matted[0][0].data.cpu().numpy()

        return matted

def main():
    model = PoseEstimator()
    src_fname = os.path.join(SRC_DIR, SRC_FNAME)

    # # MTCNN
    # model.mtcnn(src_fname) 

    # Deep3DFaceRecon 
    lm_path = os.path.join(SRC_DIR, SRC_FNAME.replace('.png', '.txt'))
    out_dict = model.modify_pose(model.estimate_pose(src_fname, lm_path))
    json_fname = os.path.join(SRC_DIR, 'camera.json')
    with open(json_fname, "w") as f:
        json.dump(out_dict, f)
    
    # Matting
    matted_im = model.matting(src_fname)
    matted_fname = src_fname.split(os.path.sep)[-1].replace('.png', '_matted.png').replace('.jpg', '_matted.png')
    save_path = os.path.join(SRC_DIR, matted_fname)
    Image.fromarray(((matted_im * 255).astype('uint8')), mode='L').save(save_path)

        

if __name__ == "__main__":
    main()