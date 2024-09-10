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
import ptwt, pywt

from functions.helpers.helpers_init import init_optimization

from functions.training.losses import SSIMLoss
from functions.helpers.meters import AverageMeter, TrackMeter, TrackMeter_testing
from functions.helpers.progress_bar import ProgressBar
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj, normalize_separate_over_ch

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html,save_modelTTT_curves

from functions.motion_simulation.motion_functions import motion_correction_NUFFT
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT

from functions.data.subsample import random_mask_partioning_3D

from functions.helpers.helpers_img_metrics import PSNR_torch


def init_modelTTT_meters():
    # !!!!
    #  Decide what quantities to track during TTT

    modelTTT_meters_per_example = {
        "modelTTT_fit_loss" : TrackMeter('decaying'),
        "modelTTT_reg_loss" : TrackMeter('decaying'),
        "modelTTT_total_fit_loss" : TrackMeter('decaying'),
        "modelTTT_selfval_loss" : TrackMeter('decaying'),
        "modelTTT_PSNR_recon_ref" : TrackMeter('increasing'),
    } 
            
    return modelTTT_meters_per_example

class UnetModelTTTModule():

    def __init__(
            self,
            args,
            model,
            final_results_dict,
            ) -> None:
        
        self.args = args
        self.model = model
        self.final_results_dict = final_results_dict

        self.modelTTT_meters_per_example = init_modelTTT_meters()

        # hold best reconstruction to save after modelTTT
        self.best_selfval_recon_img3D = None
        self.best_PSNR_recon_img3D = None

        self.final_result_dict_modelTTT = {
            "initial_PSNR_ref_vs_recon_modelTTT" : None, 
            "initial_fit_loss_modelTTT" : None, 
            "earlyStopped_PSNR_ref_vs_recon_modelTTT" : None, # refers to best_step_selfval_before_earlyStopped_modelTTT
            "earlyStopped_fit_loss_modelTTT" : None, # refers to best_step_selfval_before_earlyStopped_modelTTT
            "best_step_selfval_before_earlyStopped_modelTTT" : None, 
            "earlyStopped_at_step" : None, 
            "best_PSNR_ref_vs_recon_modelTTT" : None, # overall best psnr
            "best_step_PSNR_ref_vs_recon_modelTTT" : None, # step of overall best psnr
        }

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)

        # log gt motion parameters and estimated ones as well as error.
        logging.info(f"Gt motion parameters: {self.final_results_dict['gt_motion_params']}")
        logging.info(f"Estimated motion parameters: {self.final_results_dict['pred_motion_params_final']}")
        logging.info(f"L2 motion error gt vs. estimated: {np.sum((self.final_results_dict['gt_motion_params'] - self.final_results_dict['pred_motion_params_final'])**2)}")
        logging.info(f"TTT dc loss at estimated motion parameters: {self.final_results_dict['TTT_loss']} at step {self.final_results_dict['TTT_best_step']}")
        logging.info(f"Best L2 error in motion parameters was: {self.final_results_dict['L2_motion_parameters']}")
        logging.info(f"Best PSNR in recon was: {self.final_results_dict['PSNR_ref_vs_recon_TTT']}")

        if self.args.modelTTT_gt_motion:
            self.motion_params = torch.from_numpy(self.final_results_dict['gt_motion_params'])
            logging.info("Using gt motion parameters for modelTTT !!")
        else:
            self.motion_params = torch.from_numpy(self.final_results_dict['pred_motion_params_final'])
            logging.info("Using estimated motion parameters for modelTTT !!")
        
        self.Ns = self.motion_params.shape[0]+1
        self.motion_params.requires_grad = False

    def modelTTT(self):
        
        for name,meter in zip(self.modelTTT_meters_per_example.keys(), self.modelTTT_meters_per_example.values()):
            meter.reset()

        ###############   
        # Load k-space, sensitivity maps and mask
        filepath = os.path.join(self.args.data_drive, self.args.TTT_example_path)
        filename = filepath.split("/")[-1]

        smap_file = os.path.join(self.args.data_drive, self.args.TTT_sensmaps_path, "smaps_"+filename)
        with h5py.File(smap_file, 'r') as hf:
            smaps3D = hf['smaps'][()]
        smaps3D = torch.from_numpy(smaps3D)
        smaps3D_conj = complex_conj(smaps3D)
        binary_background_mask = torch.round(torch.sum(complex_mul(smaps3D_conj,smaps3D),0)[:,:,:,0:1])
        binary_background_mask = binary_background_mask.unsqueeze(0)
    
        with h5py.File(filepath, "r") as hf:
            ref_kspace3D = hf["kspace"][()]    
        ref_kspace3D = torch.from_numpy(ref_kspace3D)    

        with open(os.path.join(self.args.data_drive, self.args.TTT_mask_path),'rb') as fn:
            mask3D = pickle.load(fn)
            mask3D = torch.tensor(mask3D).unsqueeze(0).unsqueeze(-1) 
            logging.info(f"Using mask from {self.args.TTT_mask_path}")

        # Compute fully sampled and undersampled image volumes and load to gpu
        ref_img3D = ifft2c_ndim(ref_kspace3D, 3)
        ref_img3D = complex_mul(ref_img3D, smaps3D_conj).sum(dim=0, keepdim=False)

        # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
        # i.e. without batch dimension.
        # Batch dimensions are determined directly before passing through the network
        # and removed directly after the network output.
        ref_img3D = ref_img3D.cuda(self.args.gpu)
        smaps3D = smaps3D.cuda(self.args.gpu)
        smaps3D_conj = smaps3D_conj.cuda(self.args.gpu)
        mask3D = mask3D.cuda(self.args.gpu)
        ref_kspace3D = ref_kspace3D.cuda(self.args.gpu)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)

        ###############
        # Generate sampling trajectory
        traj = generate_interleaved_cartesian_trajectory(self.Ns, mask3D, self.args)

        ###############
        # Motion artifact simulation:
        #masked_corrupted_kspace3D = rotate_translate_3D_complex_img(ref_img3D_coil, self.gt_motion_params, traj, weight_rot=True, args=self.args)
        masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, self.motion_params, traj, weight_rot=True, args=self.args)
      
        ###############
        # Correct motion corrupted undersampled k-space with gt motion parameters
        masked_corrected_img3D = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* self.motion_params, traj, 
                                                              weight_rot=True, args=self.args, do_dcomp=self.args.modelTTT_use_nufft_with_dcomp, num_iters_dcomp=3)
        masked_corrected_img3D = complex_mul(masked_corrected_img3D, smaps3D_conj).sum(dim=0, keepdim=False)
        #masked_corrected_kspace3D = fft2c_ndim(masked_corrected_img3D_coil, 3)

        ref_kspace3D = None
        smaps3D_conj = None
        #smaps3D = None
        #ref_img3D = None
        #mask3D = None
        #masked_corrected_img3D = None
        #masked_corrupted_kspace3D = None
        #binary_background_mask = None
        #traj = None

        mask3D_fit, mask3D_selfval = random_mask_partioning_3D(mask3D, self.args.gpu)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_modelTTT)

        self.stop = False
        available_slice_nums = np.arange(masked_corrupted_kspace3D.shape[3])
        for iteration in range(self.args.num_steps_modelTTT): 
            self.model.train()

            optimizer.zero_grad()

            # Choose random slice indices for backpropagation.
            if len(available_slice_nums) < self.args.num_slices_per_grad_step_modelTTT:
                available_slice_nums = np.arange(masked_corrupted_kspace3D.shape[3])
            rec_id = np.random.choice(available_slice_nums, size=(self.args.num_slices_per_grad_step_modelTTT), replace=False)
            available_slice_nums = [x for x in available_slice_nums if x not in rec_id]

            recon_img3D = unet_forward(self.model,masked_corrected_img3D, rec_id)

            coefficient = ptwt.wavedec3(recon_img3D, pywt.Wavelet("haar"),level=1)[0]

            recon = complex_mul(recon_img3D.unsqueeze(0), smaps3D) # recon_img3D_coil
            recon = fft2c_ndim(recon, 3) # recon_kspace3D

            recon = motion_corruption_NUFFT(recon, self.motion_params, traj, weight_rot=True, args=self.args,
                                            grad_translate=True, grad_rotate=True)
            
            recon_fit = recon * mask3D_fit
            recon_selfval = recon * mask3D_selfval

            loss_fit = torch.sum(torch.abs(recon_fit-masked_corrupted_kspace3D*mask3D_fit)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_fit))
            loss_selfval = torch.sum(torch.abs(recon_selfval-masked_corrupted_kspace3D*mask3D_selfval)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_selfval))
            
            #loss_reg =torch.tensor(0)# 
            loss_reg = self.args.modelTTT_lam_recon*torch.norm(coefficient,p=1)

            total_fit_loss = loss_fit + loss_reg

            total_fit_loss.backward()
            optimizer.step()

            self.evaluate_modelTTT_step(total_fit_loss.item(), loss_reg.item(), loss_fit.item(), loss_selfval.item(), recon_img3D.detach(), recon.detach(), masked_corrected_img3D, masked_corrupted_kspace3D, ref_img3D, mask3D, mask3D_fit, mask3D_selfval, optimizer, binary_background_mask, iteration)

        self.evaluate_after_modelTTT(iteration)
        
    def evaluate_modelTTT_step(self, total_fit_loss, loss_reg, loss_fit, loss_selfval, recon_img3D, recon, masked_corrected_img3D, masked_corrupted_kspace3D, ref_img3D, mask3D, mask3D_fit, mask3D_selfval, optimizer, binary_background_mask, iteration):
        # Before modelTTT save slice images of ref, corrected and recon.
        # Also save psnr before modelTTT to final_results_dict
        # Also log num sampled indices in mask3D and fit and selfval masks
        # Update best selfval recon and best psnr recon

        self.modelTTT_meters_per_example["modelTTT_fit_loss"].update(loss_fit, iteration)
        self.modelTTT_meters_per_example["modelTTT_selfval_loss"].update(loss_selfval, iteration)
        self.modelTTT_meters_per_example["modelTTT_reg_loss"].update(loss_reg, iteration)
        self.modelTTT_meters_per_example["modelTTT_total_fit_loss"].update(total_fit_loss, iteration)

        recon_img3D = recon_img3D*binary_background_mask[0]
        psnr_ = PSNR_torch(complex_abs(recon_img3D), complex_abs(ref_img3D))
        self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].update(psnr_, iteration)

        if self.modelTTT_meters_per_example["modelTTT_selfval_loss"].best_epoch == iteration:
            self.best_selfval_recon_img3D = recon_img3D.clone()

        if self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].best_epoch == iteration:
            self.best_PSNR_recon_img3D = recon_img3D.clone()

        if iteration == 0:
            self.final_result_dict_modelTTT["initial_PSNR_ref_vs_recon_modelTTT"] = psnr_
            self.final_result_dict_modelTTT["initial_fit_loss_modelTTT"] = loss_fit

            # log num sampled indices in mask3D and fit and selfval masks
            logging.info(f"Num sampled indices in mask3D: {torch.sum(mask3D)}/{torch.numel(mask3D)} = {torch.sum(mask3D)/torch.numel(mask3D)} for shape {mask3D.shape}")
            logging.info(f"Num sampled indices in mask3D_fit: {torch.sum(mask3D_fit)}/{torch.numel(mask3D_fit)} = {torch.sum(mask3D_fit)/torch.numel(mask3D_fit)}")
            logging.info(f"Num sampled indices in mask3D_selfval: {torch.sum(mask3D_selfval)}/{torch.numel(mask3D_selfval)} = {torch.sum(mask3D_selfval)/torch.numel(mask3D_selfval)}")

            # save slice images ref, corrected and recon_initial, corrupted kspace, and predicted kspace
            list_of_slices = None
            save_slice_images_from_volume(ref_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "ref", axis_names = ["coronal","saggital","axial"])
            save_slice_images_from_volume(masked_corrected_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "corrected", axis_names = ["coronal","saggital","axial"])
            save_slice_images_from_volume(recon_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "recon_initial", axis_names = ["coronal","saggital","axial"])

            #save_slice_images_from_volume(masked_corrupted_kspace3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "masked_corrupted_ksp", axis_names = ["coronal","saggital","axial"],kspace=True,coil_index=0)
            #save_slice_images_from_volume(recon.cpu(), list_of_slices, self.args.modelTTT_results_path, "recon_initial_ksp", axis_names = ["coronal","saggital","axial"],kspace=True,coil_index=0)

        text_to_log = f"Step {iteration} | "
        for name,meter in zip(self.modelTTT_meters_per_example.keys(), self.modelTTT_meters_per_example.values()):
            text_to_log += f"{name}: {meter.val[-1]:.5f} | "

        text_to_log += f"lr: {optimizer.param_groups[0]['lr']:.5e}"
        logging.info(text_to_log)

        # early stopping
        if iteration > 3*self.args.window_size_modelTTT and not self.stop:
            if np.mean(self.modelTTT_meters_per_example['modelTTT_selfval_loss'].val[-self.args.window_size_modelTTT:]) > np.mean(self.modelTTT_meters_per_example['modelTTT_selfval_loss'].val[-2*self.args.window_size_modelTTT:-self.args.window_size_modelTTT]): 
                self.final_result_dict_modelTTT["earlyStopped_at_step"] = iteration
                self.stop = True
                # save best selfval recon volume
                # save slice images
                best_selfval_step = self.modelTTT_meters_per_example["modelTTT_selfval_loss"].best_epoch
                self.final_result_dict_modelTTT["best_step_selfval_before_earlyStopped_modelTTT"] = best_selfval_step
                self.final_result_dict_modelTTT["earlyStopped_PSNR_ref_vs_recon_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].val[best_selfval_step]
                self.final_result_dict_modelTTT["earlyStopped_fit_loss_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_fit_loss"].val[best_selfval_step]
                
                logging.info(f"Early stopping at step {iteration} with best selfval step {best_selfval_step} that exhibits PSNR {self.modelTTT_meters_per_example['modelTTT_PSNR_recon_ref'].val[best_selfval_step]}")

                list_of_slices = None
                save_slice_images_from_volume(self.best_selfval_recon_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "recon_best_selfval_beforeES", axis_names = ["coronal","saggital","axial"])
                torch.save(self.best_selfval_recon_img3D.cpu(), self.args.modelTTT_results_path+"/best_selfval_beforeES_recon.pt")
                #break

    def evaluate_after_modelTTT(self, iteration):
        # save selfval and fit curves as well as psnr curve.
        
        self.final_result_dict_modelTTT["best_PSNR_ref_vs_recon_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].best_val
        self.final_result_dict_modelTTT["best_step_PSNR_ref_vs_recon_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].best_epoch

        list_of_slices = None
        save_slice_images_from_volume(self.best_PSNR_recon_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "recon_best_PSNR_modelTTT", axis_names = ["coronal","saggital","axial"])
        
        # evaluate the possibility that there was no early stopping!
        if self.stop == False:
            self.final_result_dict_modelTTT["earlyStopped_at_step"] = iteration

            best_selfval_step = self.modelTTT_meters_per_example["modelTTT_selfval_loss"].best_epoch
            self.final_result_dict_modelTTT["best_step_selfval_before_earlyStopped_modelTTT"] = best_selfval_step
            self.final_result_dict_modelTTT["earlyStopped_PSNR_ref_vs_recon_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].val[best_selfval_step]
            self.final_result_dict_modelTTT["earlyStopped_fit_loss_modelTTT"] = self.modelTTT_meters_per_example["modelTTT_fit_loss"].val[best_selfval_step]
            
            logging.info(f"No early stopping hence stopped at final step {iteration} with best selfval step {best_selfval_step} that exhibits PSNR {self.modelTTT_meters_per_example['modelTTT_PSNR_recon_ref'].val[best_selfval_step]}")

            list_of_slices = None
            save_slice_images_from_volume(self.best_selfval_recon_img3D.cpu(), list_of_slices, self.args.modelTTT_results_path, "recon_best_selfval_beforeES", axis_names = ["coronal","saggital","axial"])
            torch.save(self.best_selfval_recon_img3D.cpu(), self.args.modelTTT_results_path+"/best_selfval_beforeES_recon.pt")
    
        save_modelTTT_curves("modelTTT_curves", self.args.modelTTT_results_path, self.modelTTT_meters_per_example, self.final_result_dict_modelTTT)
        pickle.dump( self.modelTTT_meters_per_example, open( os.path.join(self.args.modelTTT_results_path, 'modelTTT_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump( self.final_result_dict_modelTTT, open( os.path.join(self.args.modelTTT_results_path, 'final_result_dict_modelTTT.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        logging.info(f"ModelTTT finished at step {iteration} with early stopping {self.stop} at step {self.final_result_dict_modelTTT['earlyStopped_at_step']}")
        logging.info(f"Initial PSNR/fitLoss was {self.final_result_dict_modelTTT['initial_PSNR_ref_vs_recon_modelTTT']:.4f} / {self.final_result_dict_modelTTT['initial_fit_loss_modelTTT']:.5f}")
        logging.info (f"Practical PSNR/fitLoss obtained with modelTTT is {self.final_result_dict_modelTTT['earlyStopped_PSNR_ref_vs_recon_modelTTT']:.4f} / {self.final_result_dict_modelTTT['earlyStopped_fit_loss_modelTTT']:.5f} at step {self.final_result_dict_modelTTT['best_step_selfval_before_earlyStopped_modelTTT']}")
        logging.info(f"Best possible PSNR obtained with modelTTT would have been {self.final_result_dict_modelTTT['best_PSNR_ref_vs_recon_modelTTT']:.4f} at step {self.final_result_dict_modelTTT['best_step_PSNR_ref_vs_recon_modelTTT']}")
        