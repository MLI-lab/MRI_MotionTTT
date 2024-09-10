import torch
import logging
#import pickle5 as pickle
import pickle
import os
import copy
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt 

from functions.helpers.helpers_init import init_optimization

from functions.training.losses import SSIMLoss
from functions.helpers.meters import AverageMeter, TrackMeter, TrackMeter_testing
from functions.helpers.progress_bar import ProgressBar
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import motion_correction_NUFFT, generate_random_motion_params
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT


from functions.helpers.helpers_img_metrics import PSNR_torch
from torch.autograd import Variable
import ptwt, pywt

def init_l1_recon():
    # !!!!
    #  Decide what quantities to track during TTT

    l1_meters_per_example = {
        "L1_loss" : TrackMeter('decaying'),    
        "PSNR_recon_ref" : TrackMeter('increasing'),
    } 
            
    return l1_meters_per_example


class L1ReconModule():

    def __init__(
            self,
            args,
            ) -> None:
        
        self.args = args

        self.l1_meters_per_example = init_l1_recon()

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)

    def l1_recon(self):

        # self.model.eval()
        for name,meter in zip(self.l1_meters_per_example.keys(), self.l1_meters_per_example.values()):
            meter.reset()

        ###############   
        # Load k-space, sensitivity maps and mask
        filepath = os.path.join(self.args.data_drive, self.args.TTT_example_path)
        filename = filepath.split("/")[-1]
        volume_name = filename.split(".")[0]

        smap_file = os.path.join(self.args.data_drive, self.args.TTT_sensmaps_path, "smaps_"+filename)
        with h5py.File(smap_file, 'r') as hf:
            smaps3D = hf['smaps'][()]
        smaps3D = torch.from_numpy(smaps3D)
        smaps3D_conj = complex_conj(smaps3D)
        binary_background_mask = torch.round(torch.sum(complex_mul(smaps3D_conj,smaps3D),0)[:,:,:,0:1])
        binary_background_mask = binary_background_mask.unsqueeze(0)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)
    
        with h5py.File(filepath, "r") as hf:
            ref_kspace3D = hf["kspace"][()]    
        ref_kspace3D = torch.from_numpy(ref_kspace3D)    

        with open(os.path.join(self.args.data_drive, self.args.TTT_mask_path),'rb') as fn:
            mask3D = pickle.load(fn)
            mask3D = torch.tensor(mask3D).unsqueeze(0).unsqueeze(-1) 
            logging.info(f"Using mask from {self.args.TTT_mask_path}")

        # Compute fully sampled and undersampled image volumes and load to gpu
        ref_img3D_coil = ifft2c_ndim(ref_kspace3D, 3)
        ref_img3D = complex_mul(ref_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        masked_kspace3D = ref_kspace3D * mask3D
        masked_img3D_coil = ifft2c_ndim(masked_kspace3D, 3)
        masked_img3D = complex_mul(masked_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
        # i.e. without batch dimension.
        # Batch dimensions are determined directly before passing through the network
        # and removed directly after the network output.
        masked_img3D = masked_img3D.cuda(self.args.gpu)
        ref_img3D = ref_img3D.cuda(self.args.gpu)
        #ref_img3D_coil = ref_img3D_coil.cuda(self.args.gpu)
        smaps3D = smaps3D.cuda(self.args.gpu)
        smaps3D_conj = smaps3D_conj.cuda(self.args.gpu)
        mask3D = mask3D.cuda(self.args.gpu)
        ref_kspace3D = ref_kspace3D.cuda(self.args.gpu)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)

        ###############
        # Generate sampling trajectory
        traj = generate_interleaved_cartesian_trajectory(self.args.Ns, mask3D, self.args)
                
        ###############
        # Generate Motion State
        self.gt_motion_params = generate_random_motion_params(self.args.Ns-1, self.args.max_trans, self.args.max_rot, self.args.random_motion_seed).cuda(self.args.gpu)

        ###############
        # Motion artifact simulation:
        if self.args.max_trans == 0 and self.args.max_rot == 0:
            masked_kspace3D = masked_kspace3D.cuda(self.args.gpu)

        masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, self.gt_motion_params, traj, weight_rot=True, args=self.args)
        
        ref_kspace3D = None
        ###############
        # Init Reconstruction Volume
        mse = torch.nn.MSELoss()
        recon = Variable(torch.zeros(ref_img3D.shape)).cuda(self.args.gpu)
        recon.data.uniform_(0,1)
        recon.requires_grad = True
        
        optimizer = torch.optim.SGD([recon],lr=self.args.lr_l1_recon)
        best_psnr = 0

        logging.info(f"Starting L1 Reconstruction with {self.args.num_steps_l1_recon} steps, lr {self.args.lr_l1_recon} and lambda {self.args.lam_l1_recon}.")
        
        for iteration in range(self.args.num_steps_l1_recon):
            optimizer.zero_grad()
            
            # Step 1: Apply forward model
            # a. Expand opeartor:
            # recon = recon*scal_factor
            recon_coil = complex_mul(recon.unsqueeze(0), smaps3D)
            # b. Apply Mask:
            recon_kspace3d_coil = fft2c_ndim(recon_coil, 3)

            if self.args.max_trans == 0 and self.args.max_rot == 0:
                recon_kspace3d_coil = recon_kspace3d_coil*mask3D
            else:
                recon_kspace3d_coil = motion_corruption_NUFFT(recon_kspace3d_coil, self.gt_motion_params, traj, weight_rot=True, args=self.args)
            # c. Scale the kspace:
            # with torch.no_grad():
            #     scale_factor = torch.sqrt(torch.sum(masked_corrupted_kspace3D**2))/torch.sqrt(torch.sum(recon_kspace3d_coil**2))

            scale_factor = torch.sqrt(torch.sum(masked_corrupted_kspace3D**2))/torch.sqrt(torch.sum(recon_kspace3d_coil**2))

            recon_kspace3d_coil = recon_kspace3d_coil*scale_factor
            
            # Step 2: Calculating the loss and backward
            # a. take wavelet of reconstruction
            coefficient = ptwt.wavedec3(recon*scale_factor, pywt.Wavelet("haar"),level=1)[0]
            # b. Calculating the loss and backward
            if self.args.max_trans == 0 and self.args.max_rot == 0:
                loss = mse( recon_kspace3d_coil , masked_kspace3D ) + self.args.lam_l1_recon*torch.norm(coefficient,p=1)
            else:
                loss = mse( recon_kspace3d_coil , masked_corrupted_kspace3D ) + self.args.lam_l1_recon*torch.norm(coefficient,p=1)
            loss.backward()
            optimizer.step()
            psnr_step = PSNR_torch(complex_abs(recon*scale_factor*binary_background_mask), complex_abs(ref_img3D))
            
            ### Log the data
            logging.info("iteration{} -- fitting loss: {}|PSNR: {}".format(iteration,loss.data,psnr_step))
            
            ### Save reconstruction per 100 iterations:
            if psnr_step>best_psnr:
                best_psnr = psnr_step

                torch.save(recon.detach()*scale_factor.detach(), self.args.l1_results_path+"/l1_reconstruction.pt")
        logging.info("Best PSNR: {}".format(best_psnr))
        # torch.save(recon, self.args.l1_results_path+"/l1_reconstruction.pt")
        
        self.evaluate_after_l1_recon(binary_background_mask, ref_img3D)

        
    def evaluate_after_l1_recon(self, binary_background_mask, ref_img3D):

        # load the best reconstruction
        recon = torch.load(self.args.l1_results_path+"/l1_reconstruction.pt")
        recon = recon.cuda(self.args.gpu)
        recon = recon * binary_background_mask
        psnr = PSNR_torch(complex_abs(recon), complex_abs(ref_img3D))
        logging.info(f"PSNR of the best reconstruction: {psnr}")

        list_of_slices = None
        save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.l1_results_path, "recon_axial_l1", axis_names = ["coronal","saggital","axial"])

    
            