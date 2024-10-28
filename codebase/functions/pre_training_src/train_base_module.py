
import numpy as np
import torch
import logging
import os
import pickle
import argparse

from functions.utils.helpers.meters import AverageMeter, TrackMeter, TrackMeter_testing
from functions.utils.helpers.progress_bar import ProgressBar
from functions.pre_training_src.losses import SSIMLoss
from functions.utils.helpers.helpers_log_save_image_utils import add_img_to_tensorboard, save_slice_images_from_volume
from functions.utils.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim

from functions.utils.motion_simulation.motion_helpers import align_3D_volumes
from functions.utils.motion_simulation.motion_forward_backward_models import motion_correction_NUFFT, motion_corruption_NUFFT
from functions.utils.motion_simulation.motion_trajectories import sim_motion_get_gt_motion_traj
from functions.utils.motion_simulation.sampling_trajectories import sim_motion_get_traj


class TrainModuleBase():
    def __init__(
            self,
            args,
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            train_loss_function,
            tb_writer
        ) -> None:
        
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss_function = train_loss_function
        self.tb_writer = tb_writer

        self.train_meters_per_epoch, self.train_meters_over_epochs, self.val_meters_per_epoch, self.val_meters_over_epochs = self.init_train_val_meters()

        # Store validation psnrs per volume
        self.psnrs = []

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)

        self.val_motion_ind = 0


    def init_train_val_meters(self):

        train_meters_per_epoch = {
            'train_loss': AverageMeter(), 
            'train_loss_img': AverageMeter(), 
            'train_loss_ksp': AverageMeter()
            }
        train_meters_over_epochs = {
            'train_loss': TrackMeter('decaying'), 
            'train_loss_img': TrackMeter('decaying'), 
            'train_loss_ksp': TrackMeter('decaying')
            }
        

        val_meters_per_epoch = {
            'PSNR' : AverageMeter(),
            'PSNR_corrected' : AverageMeter(),
            'PSNR_corrupted' : AverageMeter(),
            }
        val_meters_over_epochs = { 
            'PSNR' : TrackMeter('increasing'),
            'PSNR_corrected' : TrackMeter('increasing'),
            'PSNR_corrupted' : TrackMeter('increasing'),
            } 
        
        return train_meters_per_epoch, train_meters_over_epochs, val_meters_per_epoch, val_meters_over_epochs
    
    def train_epoch(self, epoch):
        self.model.train()
        train_bar = ProgressBar(self.train_loader, epoch)
        for meter in self.train_meters_per_epoch.values():
            meter.reset()

        
        for batch_id, batch in enumerate(train_bar):

            target_kspace_3D, input_kspace_3D, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename, _ = batch

            target_kspace_3D = target_kspace_3D[0].cuda(self.args.gpu)                                       # (coils, x, y, z, ch)
            input_kspace_3D = input_kspace_3D[0].cuda(self.args.gpu)                                         # (coils, x, y, z, ch)
            binary_background_mask_3D = binary_background_mask_3D[0].cuda(self.args.gpu)                     # (x, y, z, 1)
            sens_maps_3D = sens_maps_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
            sens_maps_conj_3D = sens_maps_conj_3D[0].cuda(self.args.gpu)                                     # (coils, x, y, z, ch)
            target_img_3D = target_img_3D[0].cuda(self.args.gpu)                                             # (x, y, z, ch)
            mask3D = mask3D[0].cuda(self.args.gpu)                                                           # (1, x, y, z, 1)

            # Get sampling trajectory for the current mask the type of sampling traj
            # only matters if train_on_motion_corrected_inputs or train_on_motion_corrupted_inputs is true.
            traj = sim_motion_get_traj(self.args, mask3D, verbose=False)

            if self.args.train_on_motion_free_inputs:
                # Obtain pairs of motion-free undersampled inputs and fully sampled targets
                if self.args.train_use_nufft_adjoint:
                    # Apply adjoint nufft with all zero motion parameters to obtain the motion-free coil images
                    input_nufft_img3D_coil = motion_correction_NUFFT(input_kspace_3D, None, traj, weight_rot=True, args=self.args,
                                                                                do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                    input_img_3D = complex_mul(input_nufft_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    input_img_3D = complex_mul(ifft2c_ndim(input_kspace_3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)
            else:
                input_img_3D = None
            
            if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                # In addition to the motion-free input image a motion corrupted OR motion corrected
                # input image can be used for training.
                assert isinstance(self.args.train_Ns, list) and isinstance(self.args.train_max_rots, list) and isinstance(self.args.train_max_trans, list)
                if self.args.train_on_motion_corrected_inputs and self.args.train_on_motion_corrupted_inputs:
                    raise ValueError("Can only train on either motion corrected or corrupted inputs, not both.")

                # Generate ground truth Motion trajectory. Both gt_motion_params and gt_traj have per k-space line resolution
                # The random motion seed depends on the filename and the random motion seed list
                # which determines how many different motion trajectories are generated for a given
                # trainings volume and level of motion severity.
                self.args.max_rot = np.random.choice(self.args.train_max_mot)
                self.args.max_trans = self.args.max_rot
                self.args.num_motion_events = np.random.choice(self.args.train_num_motion_events)
                self.args.num_intraShot_events = int(np.ceil(self.args.num_motion_events/2))
                # Extract all numbers from the filename and use them as seed (designed for the Calgary-Campinas dataset)
                seed = filename[0].split('.')[0].split('_')[0]
                seed = [int(s) for s in seed if s.isdigit()]
                seed.append(np.random.choice(self.args.train_random_motion_seeds))	
                seed = int("".join(map(str, seed)))
                self.args.random_motion_seed = seed

                gt_motion_params, gt_traj, intraShot_event_inds = sim_motion_get_gt_motion_traj(self.args, traj, verbose=False)

                # Motion artifact simulation:
                # Reduce the number of motion states by combining motion states with the same motion parameters to save some time here
                gt_motion_params_combined = gt_motion_params[0:1,:]
                gt_traj_combined = ([gt_traj[0][0]], [gt_traj[1][0]])
                for i in range(1, gt_motion_params.shape[0]):
                    if torch.sum(torch.abs(gt_motion_params[i]-gt_motion_params[i-1])) > 0:
                        gt_motion_params_combined = torch.cat((gt_motion_params_combined, gt_motion_params[i:i+1,:]), dim=0)
                        gt_traj_combined[0].append(gt_traj[0][i]) 
                        gt_traj_combined[1].append(gt_traj[1][i])
                    else:
                        gt_traj_combined[0][-1] = np.concatenate((gt_traj_combined[0][-1], gt_traj[0][i]), axis=0)
                        gt_traj_combined[1][-1] = np.concatenate((gt_traj_combined[1][-1], gt_traj[1][i]), axis=0)

                target_img_3D_coil = ifft2c_ndim(target_kspace_3D, 3)
                input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, target_img_3D_coil, gt_motion_params_combined, gt_traj_combined, weight_rot=True, args=self.args,
                                                                    max_coil_size=self.args.train_nufft_max_coil_size)
                
                if self.args.train_on_motion_corrected_inputs:
                    # Correct motion corrupted undersampled k-space with gt motion parameters
                    input_corrected_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params_combined, gt_traj_combined, 
                                                                        weight_rot=True, args=self.args, do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                        max_coil_size=self.args.train_nufft_max_coil_size)
                    input_cor_img3D = complex_mul(input_corrected_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    if self.args.train_use_nufft_adjoint:
                        # Apply adjoint nufft with all zero motion parameters to obtain the corrupted coil images
                        input_corrupted_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, None, gt_traj_combined, weight_rot=True, args=self.args,
                                                                            do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                            max_coil_size=self.args.train_nufft_max_coil_size)
                        input_cor_img3D = complex_mul(input_corrupted_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    else:
                        input_cor_img3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)
            else:
                input_cor_img3D = None

            if self.args.train_always_on_mild_motion:
                # We provide the option to always train on mild motion to bias the
                # the reconstruction abilities of the end-to-end motion correction network
                # towards a regime in which the results are still somewhat acceptable.
                self.args.max_rot = 2
                self.args.max_trans = 2
                self.args.num_motion_events = 1
                self.args.num_intraShot_events = 1
                # Extract all numbers from the filename and use them as seed
                seed = filename[0].split('.')[0].split('_')[0]
                seed = [int(s) for s in seed if s.isdigit()]
                seed.append(np.random.choice([6,7,8,9])) # provide a fixed set of seeds for mild motion
                seed = int("".join(map(str, seed)))
                self.args.random_motion_seed = seed

                gt_motion_params, gt_traj, _ = sim_motion_get_gt_motion_traj(self.args, traj, verbose=False)

                # Motion artifact simulation:
                # Reduce the number of motion states by combining motion states with the same motion parameters to save some time here
                gt_motion_params_combined = gt_motion_params[0:1,:]
                gt_traj_combined = ([gt_traj[0][0]], [gt_traj[1][0]])
                for i in range(1, gt_motion_params.shape[0]):
                    if torch.sum(torch.abs(gt_motion_params[i]-gt_motion_params[i-1])) > 0:
                        gt_motion_params_combined = torch.cat((gt_motion_params_combined, gt_motion_params[i:i+1,:]), dim=0)
                        gt_traj_combined[0].append(gt_traj[0][i]) 
                        gt_traj_combined[1].append(gt_traj[1][i])
                    else:
                        gt_traj_combined[0][-1] = np.concatenate((gt_traj_combined[0][-1], gt_traj[0][i]), axis=0)
                        gt_traj_combined[1][-1] = np.concatenate((gt_traj_combined[1][-1], gt_traj[1][i]), axis=0)

                target_img_3D_coil = ifft2c_ndim(target_kspace_3D, 3)
                input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, target_img_3D_coil, gt_motion_params_combined, gt_traj_combined, weight_rot=True, args=self.args,
                                                                    max_coil_size=self.args.train_nufft_max_coil_size)
                
                if self.args.train_on_motion_corrected_inputs:
                    # Correct motion corrupted undersampled k-space with gt motion parameters
                    input_corrected_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params_combined, gt_traj_combined, 
                                                                        weight_rot=True, args=self.args, do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                        max_coil_size=self.args.train_nufft_max_coil_size)
                    input_corMild_img3D = complex_mul(input_corrected_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    if self.args.train_use_nufft_adjoint:
                        # Apply adjoint nufft with all zero motion parameters to obtain the corrupted coil images
                        input_corrupted_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, None, gt_traj_combined, weight_rot=True, args=self.args,
                                                                            do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                            max_coil_size=self.args.train_nufft_max_coil_size)
                        input_corMild_img3D = complex_mul(input_corrupted_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    else:
                        input_corMild_img3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    
            else:
                input_corMild_img3D = None


            # train_batch_size_per_axis is a list of batch sizes for each axis.
            # E.g. [None, 20, 20] means that we obtain 20 slices from the second and third axis
            for ax_ind, batch_size in enumerate(self.args.train_batch_size_per_axis):
                if batch_size:
                    # move corresponding axis to batch dimension
                    sens_maps_3D = sens_maps_3D.moveaxis(ax_ind+1, 0)
                    target_img_3D = target_img_3D.moveaxis(ax_ind, 0)
                    binary_background_mask_3D = binary_background_mask_3D.moveaxis(ax_ind, 0)

                    if self.args.train_on_motion_free_inputs:
                        input_img_3D = input_img_3D.moveaxis(ax_ind, 0)
                        # randomly select slices
                        rec_id = np.random.choice(range(input_img_3D.shape[0]),size=(batch_size), replace=False)
                    else:
                        rec_id = None

                    if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                        input_cor_img3D = input_cor_img3D.moveaxis(ax_ind, 0)
                        rec_id_cor = np.random.choice(range(input_cor_img3D.shape[0]),size=(batch_size), replace=False)
                    else:
                        rec_id_cor = None

                    if self.args.train_always_on_mild_motion:
                        input_corMild_img3D = input_corMild_img3D.moveaxis(ax_ind, 0)
                        rec_id_corMild = np.random.choice(range(input_corMild_img3D.shape[0]),size=(batch_size), replace=False)
                    else:
                        rec_id_corMild = None
                        
                    recon_image_fg_1c_last_batch, target_image_2D_last_batch, input_img_2D_last_batch, loss, loss_img, loss_ksp = self.train_step(input_img_3D, input_cor_img3D, input_corMild_img3D, sens_maps_3D, target_img_3D, target_kspace_3D, binary_background_mask_3D, rec_id, rec_id_cor, rec_id_corMild, ax_ind)
                    name_tag = f"_ax{ax_ind}"
                    self.train_step_logging(epoch, batch_id, input_img_2D_last_batch[0], target_image_2D_last_batch[0], recon_image_fg_1c_last_batch[0], loss, loss_img, loss_ksp, name_tag)

                    # move axis back to original position
                    if self.args.train_on_motion_free_inputs:
                        input_img_3D = input_img_3D.moveaxis(0, ax_ind)
                    sens_maps_3D = sens_maps_3D.moveaxis(0, ax_ind+1)
                    target_img_3D = target_img_3D.moveaxis(0, ax_ind)
                    binary_background_mask_3D = binary_background_mask_3D.moveaxis(0, ax_ind)

                    if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                        input_cor_img3D = input_cor_img3D.moveaxis(0, ax_ind)

                    if self.args.train_always_on_mild_motion:
                        input_corMild_img3D = input_corMild_img3D.moveaxis(0, ax_ind)

        self.train_epoch_logging(epoch)
        if self.scheduler:
            self.scheduler.step()
                    


    def train_step_logging(self, epoch, batch_id, inputs_img_full, target_image_fg, recon_image_fg, loss, loss_img, loss_ksp, name_tag=""):

        self.train_meters_per_epoch["train_loss"].update(loss)
        self.train_meters_per_epoch["train_loss_img"].update(loss_img)
        self.train_meters_per_epoch["train_loss_ksp"].update(loss_ksp)

        if (epoch % self.args.log_imgs_to_tb_every == 0 or epoch in [0,1,2,3,4] ) and batch_id in [0] and self.tb_writer:
            add_img_to_tensorboard(self.tb_writer, epoch, f"train_vis_{batch_id}"+name_tag, inputs_img_full.unsqueeze(0), target_image_fg.unsqueeze(0), recon_image_fg.unsqueeze(0), self.ssim_loss)

    def train_epoch_logging(self, epoch):
        self.train_meters_over_epochs["train_loss"].update(self.train_meters_per_epoch["train_loss"].avg, epoch)
        self.train_meters_over_epochs["train_loss_img"].update(self.train_meters_per_epoch["train_loss_img"].avg, epoch)
        self.train_meters_over_epochs["train_loss_ksp"].update(self.train_meters_per_epoch["train_loss_ksp"].avg, epoch)

        if self.tb_writer:
            self.tb_writer.add_scalar("Train loss total", self.train_meters_per_epoch["train_loss"].avg, epoch)
            self.tb_writer.add_scalar("Train loss img", self.train_meters_per_epoch["train_loss_img"].avg, epoch)
            self.tb_writer.add_scalar("Train loss ksp", self.train_meters_per_epoch["train_loss_ksp"].avg, epoch)


    def val_epoch(self, epoch):
        self.model.eval()
        val_bar = ProgressBar(self.val_loader, epoch)
        for meter in self.val_meters_per_epoch.values():
            meter.reset()

        with torch.no_grad():
            self.psnrs = []
            for batch_id, batch in enumerate(val_bar):

                target_kspace_3D, input_kspace_3D, binary_background_mask_3D, _, sens_maps_conj_3D, target_img_3D, mask3D, filename, random_motion_seeds = batch

                target_kspace_3D = target_kspace_3D[0].cuda(self.args.gpu)                                       # (coils, x, y, z, ch)
                input_kspace_3D = input_kspace_3D[0].cuda(self.args.gpu)                                         # (coils, x, y, z, ch)
                binary_background_mask_3D = binary_background_mask_3D[0].cuda(self.args.gpu)                     # (x, y, z, 1)
                sens_maps_conj_3D = sens_maps_conj_3D[0].cuda(self.args.gpu)                                     # (coils, x, y, z, ch)
                target_img_3D = target_img_3D[0].cuda(self.args.gpu)                                             # (x, y, z, ch)
                mask3D = mask3D[0].cuda(self.args.gpu)                                                           # (1, x, y, z, 1)


                # Get sampling trajectory for the current mask. The type of sampling traj
                # only matters if train_on_motion_corrected_inputs or train_on_motion_corrupted_inputs is true.
                traj = sim_motion_get_traj(self.args, mask3D, verbose=False)

                # The motion-free input image is always obtained and used for validation
                if self.args.train_use_nufft_adjoint:
                    # Apply adjoint nufft with all zero motion parameters to obtain the motion-free coil images
                    input_img_3D = motion_correction_NUFFT(input_kspace_3D, None, traj, weight_rot=True, args=self.args,
                                                                                do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                    input_img_3D = complex_mul(input_img_3D, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    input_img_3D = complex_mul(ifft2c_ndim(input_kspace_3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)

                # In addition to the motion-free input image a motion corrupted OR motion corrected
                # input image is used for validation.
                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    assert isinstance(self.args.train_Ns, list) and isinstance(self.args.train_max_rots, list) and isinstance(self.args.train_max_trans, list)
                    if self.args.train_on_motion_corrected_inputs and self.args.train_on_motion_corrupted_inputs:
                        raise ValueError("Can only validate on either motion corrected or corrupted inputs, not both.")

                    # Generate ground truth Motion trajectory. Both gt_motion_params and gt_traj have per k-space line resolution
                    # The random motion seed is given from the validation set
                    # in every other validation epoch the motion severity is switched between two (or more) levels
                    self.args.max_rot = self.args.val_max_mot[self.val_motion_ind]
                    self.args.max_trans = self.args.val_max_mot[self.val_motion_ind]
                    self.args.num_motion_events = self.args.val_num_motion_events[self.val_motion_ind]
                    self.args.num_intraShot_events = int(np.ceil(self.args.num_motion_events/2))
                    self.args.random_motion_seed = random_motion_seeds['seed1']

                    gt_motion_params, gt_traj, intraShot_event_inds = sim_motion_get_gt_motion_traj(self.args, traj, verbose=False)

                    # Motion artifact simulation:
                    # Reduce the number of motion states by combining motion states with the same motion parameters to save some time here
                    gt_motion_params_combined = gt_motion_params[0:1,:]
                    gt_traj_combined = ([gt_traj[0][0]], [gt_traj[1][0]])
                    for i in range(1, gt_motion_params.shape[0]):
                        if torch.sum(torch.abs(gt_motion_params[i]-gt_motion_params[i-1])) > 0:
                            gt_motion_params_combined = torch.cat((gt_motion_params_combined, gt_motion_params[i:i+1,:]), dim=0)
                            gt_traj_combined[0].append(gt_traj[0][i]) 
                            gt_traj_combined[1].append(gt_traj[1][i])
                        else:
                            gt_traj_combined[0][-1] = np.concatenate((gt_traj_combined[0][-1], gt_traj[0][i]), axis=0)
                            gt_traj_combined[1][-1] = np.concatenate((gt_traj_combined[1][-1], gt_traj[1][i]), axis=0)

                    target_img_3D_coil = ifft2c_ndim(target_kspace_3D, 3)
                    input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, target_img_3D_coil, gt_motion_params_combined, gt_traj_combined, weight_rot=True, args=self.args,
                                                                        max_coil_size=self.args.train_nufft_max_coil_size)
                    
                    if self.args.train_on_motion_corrected_inputs:
                        # Correct motion corrupted undersampled k-space with gt motion parameters
                        input_cor_img3D = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params_combined, gt_traj_combined, 
                                                                            weight_rot=True, args=self.args, do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                            max_coil_size=self.args.train_nufft_max_coil_size)
                        input_cor_img3D = complex_mul(input_cor_img3D, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    else:
                        if self.args.train_use_nufft_adjoint:
                            # Apply adjoint nufft with all zero motion parameters to obtain the corrupted coil images
                            input_cor_img3D = motion_correction_NUFFT(input_corrupted_kspace3D, None, gt_traj_combined, weight_rot=True, args=self.args,
                                                                                do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                                max_coil_size=self.args.train_nufft_max_coil_size)
                            input_cor_img3D = complex_mul(input_cor_img3D, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                        else:
                            input_cor_img3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    input_cor_img3D = None

                sens_maps_conj_3D = None
                input_kspace_3D = None
                target_kspace_3D = None
                input_corrupted_kspace3D = None
                target_img_3D_coil = None
                input_nufft_img3D_coil = None
                input_corrupted_img3D_coil = None
                input_corrected_img3D_coil = None


                # move corresponding axis to batch dimension
                ax_ind = 2 # here we validate over axis 2 (axial in our case)
                input_img_2D = input_img_3D.moveaxis(ax_ind, 0)
                target_image_2D = target_img_3D.moveaxis(ax_ind, 0)
                binary_background_mask_2D = binary_background_mask_3D.moveaxis(ax_ind, 0)
                
                target_image_fg_2c = target_image_2D * binary_background_mask_2D                          # (batch, x, y, ch)
                target_image_fg_1c = complex_abs(target_image_fg_2c)#.unsqueeze(1)                         # (batch, 1, x, y)

                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    input_cor_img_full_2D = input_cor_img3D.moveaxis(ax_ind, 0)

                recon_image_fg_1c = self.val_step(input_img_2D, binary_background_mask_2D)

                psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)

                self.val_meters_per_epoch["PSNR"].update(psnr_sense.item())
                self.psnrs.append(psnr_sense.item())

                if (epoch % self.args.log_imgs_to_tb_every == 0 or epoch in [0,1,2,3,4] ) and batch_id in [0,1] and self.tb_writer:
                    slice_num = input_img_2D.shape[0]//2
                    add_img_to_tensorboard(self.tb_writer, epoch, f"val_motfree_{filename[0].split('.')[0]}_ax{ax_ind}_s{slice_num}", input_img_2D[slice_num:slice_num+1], target_image_fg_2c[slice_num:slice_num+1], recon_image_fg_1c[slice_num:slice_num+1], self.ssim_loss)

                    slice_num = input_img_2D.shape[2]//2
                    add_img_to_tensorboard(self.tb_writer, epoch, f"val_motfree_{filename[0].split('.')[0]}_ax1_s{slice_num}", input_img_2D[:,:,slice_num].unsqueeze(0), target_image_fg_2c[:,:,slice_num].unsqueeze(0), recon_image_fg_1c[:,:,slice_num].unsqueeze(0), self.ssim_loss)
            
                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    recon_image_fg_1c = self.val_step(input_cor_img_full_2D, binary_background_mask_2D)

                    psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                    psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)

                    if self.args.train_on_motion_corrected_inputs:
                        self.val_meters_per_epoch["PSNR_corrected"].update(psnr_sense.item())
                    elif self.args.train_on_motion_corrupted_inputs:
                        self.val_meters_per_epoch["PSNR_corrupted"].update(psnr_sense.item())

                    if (epoch % self.args.log_imgs_to_tb_every == 0 or epoch in [0,1,2,3,4] ) and batch_id in [0,1] and self.tb_writer:
                        slice_num = input_img_2D.shape[0]//2
                        add_img_to_tensorboard(self.tb_writer, epoch, f"val_cor_{filename[0].split('.')[0]}_ax{ax_ind}_s{slice_num}", input_cor_img_full_2D[slice_num:slice_num+1], target_image_fg_2c[slice_num:slice_num+1], recon_image_fg_1c[slice_num:slice_num+1], self.ssim_loss)

                        slice_num = input_img_2D.shape[2]//2
                        add_img_to_tensorboard(self.tb_writer, epoch, f"val_cor_{filename[0].split('.')[0]}_ax1_s{slice_num}", input_cor_img_full_2D[:,:,slice_num].unsqueeze(0), target_image_fg_2c[:,:,slice_num].unsqueeze(0), recon_image_fg_1c[:,:,slice_num].unsqueeze(0), self.ssim_loss)
        
            for name,meter in zip(self.val_meters_per_epoch.keys(), self.val_meters_per_epoch.values()):
                self.val_meters_over_epochs[name].update(meter.avg, epoch)
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"Val {name}", meter.avg, epoch)

            if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                self.val_motion_ind = (self.val_motion_ind + 1) % len(self.args.val_max_mot) 


    def log_after_epoch(self, epoch):
        text_to_log = f"Epoch {epoch} | "
        for name,meter in zip(self.train_meters_per_epoch.keys(), self.train_meters_per_epoch.values()):
            text_to_log += f"{name}: {meter.avg:.5f} | "
        
        text_to_log += f"lr: {self.optimizer.param_groups[0]['lr']:.5e} | "

        if epoch % self.args.val_every == 0:
            for name,meter in zip(self.val_meters_per_epoch.keys(), self.val_meters_per_epoch.values()):
                text_to_log += f"val {name}: {meter.avg:.5f} | "

        self.psnrs = [round(psnr, 3) for psnr in self.psnrs]
        
        if epoch % self.args.val_every == 0:
            text_to_log += f"PSNRs per val vol (mot free): {self.psnrs} | Max mot: {self.args.max_rot} | NumEvents: {self.args.num_motion_events}"
        logging.info(text_to_log)

    def save_checkpoints(self, epoch, save_metrics):

        for save_metric in save_metrics:
            if self.val_meters_over_epochs[save_metric].best_epoch == epoch:

                state_dict = {
                    "epoch": epoch,
                    "val_score": self.val_meters_over_epochs[save_metric].val[-1],
                    "best_epoch": self.val_meters_over_epochs[save_metric].best_epoch,
                    "best_score": self.val_meters_over_epochs[save_metric].best_val,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "args": argparse.Namespace(**{k: v for k, v in vars(self.args).items() if not callable(v)}),
                }
                torch.save(state_dict, os.path.join(self.args.checkpoint_path , f"checkpoint_best_{save_metric}.pt"))

        if epoch % self.args.save_checkpoints_every == 0 and epoch != 0:
            torch.save((self.model.state_dict()), os.path.join(self.args.checkpoint_path, f"checkpoint_epoch_{epoch}.pt"))
            state_dict = {
                    "epoch": epoch,
                    "val_score": self.val_meters_over_epochs[save_metric].val[-1],
                    "best_epoch": self.val_meters_over_epochs[save_metric].best_epoch,
                    "best_score": self.val_meters_over_epochs[save_metric].best_val,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "args": argparse.Namespace(**{k: v for k, v in vars(self.args).items() if not callable(v)}),
                }
            torch.save(state_dict, os.path.join(self.args.checkpoint_path , f"checkpoint_epoch_{epoch}.pt"))
            


    def log_and_save_final_metrics(self, epoch):
        for name,meter in zip(self.val_meters_over_epochs.keys(), self.val_meters_over_epochs.values()):
            logging.info(f"Done training after {epoch} epochs! Best val {name} score of {meter.best_val:.5f} obtained after epoch {meter.best_epoch}.")

        pickle.dump( self.val_meters_over_epochs, open(os.path.join(self.args.train_results_path, 'valid_tracks_metrics.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump( self.train_meters_over_epochs, open(os.path.join(self.args.train_results_path, 'train_tracks_metrics.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        

    def eval_valset(self, mode, ax_ind, save_name, save_recon, maxmot=None, num_events=None, seed=None):
        self.model.eval()
        val_bar = ProgressBar(self.val_loader, epoch=0)

        save_path = os.path.join(self.args.train_results_path, f"test_results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        save_path = os.path.join(save_path, f"test_perf_{save_name}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if save_recon:
            save_path_recon = os.path.join(save_path, "reconstructions")
            if not os.path.exists(save_path_recon):
                os.makedirs(save_path_recon)

        val_meters = {
            'PSNR_unet' : TrackMeter_testing(),
            'PSNR_unet_aligned' : TrackMeter_testing(),
            'PSNR_zerofilled' : TrackMeter_testing(),
            }

        for meter in val_meters.values():
            meter.reset()

        with torch.no_grad():
            for batch_id, batch in enumerate(val_bar):

                target_kspace_3D, input_kspace_3D, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename, random_motion_seeds = batch

                target_kspace_3D = target_kspace_3D[0].cuda(self.args.gpu)                                       # (coils, x, y, z, ch)
                input_kspace_3D = input_kspace_3D[0].cuda(self.args.gpu)                                         # (coils, x, y, z, ch)
                binary_background_mask_3D = binary_background_mask_3D[0].cuda(self.args.gpu)                     # (x, y, z, 1)
                sens_maps_3D = sens_maps_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
                sens_maps_conj_3D = sens_maps_conj_3D[0].cuda(self.args.gpu)                                     # (coils, x, y, z, ch)
                target_img_3D = target_img_3D[0].cuda(self.args.gpu)                                             # (x, y, z, ch)
                mask3D = mask3D[0].cuda(self.args.gpu)                                                           # (1, x, y, z, 1)

                target_image_2D = target_img_3D.moveaxis(ax_ind, 0)
                binary_background_mask_2D = binary_background_mask_3D.moveaxis(ax_ind, 0)
                
                target_image_fg_2c = target_image_2D * binary_background_mask_2D                          # (batch, x, y, ch)
                target_image_fg_1c = complex_abs(target_image_fg_2c)                                      # (batch, 1, x, y)


                # Get sampling trajectory for the current mask
                traj = sim_motion_get_traj(self.args, mask3D, verbose=False)

                if mode == 'motion_free':
                    if self.args.train_use_nufft_adjoint:
                        # Apply adjoint nufft with all zero motion parameters to obtain the motion-free coil images
                        input_nufft_img3D_coil = motion_correction_NUFFT(input_kspace_3D, None, traj, weight_rot=True, args=self.args,
                                                                                    do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                        input_img_3D = complex_mul(input_nufft_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    else:
                        input_img_3D = complex_mul(ifft2c_ndim(input_kspace_3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)

                elif mode == 'motion_corrupted' or mode == 'motion_corrected':
                    assert maxmot is not None and num_events is not None and seed is not None

                    # Generate ground truth Motion trajectory. Both gt_motion_params and gt_traj have per k-space line resolution
                    self.args.max_rot = maxmot
                    self.args.max_trans = maxmot
                    self.args.num_motion_events = num_events
                    self.args.num_intraShot_events = int(np.ceil(num_events/2))
                    self.args.random_motion_seed = random_motion_seeds[seed]

                    gt_motion_params, gt_traj, intraShot_event_inds = sim_motion_get_gt_motion_traj(self.args, traj, verbose=False)

                    # Motion artifact simulation:
                    # Reduce the number of motion states by combining motion states with the same motion parameters to save some time here
                    gt_motion_params_combined = gt_motion_params[0:1,:]
                    gt_traj_combined = ([gt_traj[0][0]], [gt_traj[1][0]])
                    for i in range(1, gt_motion_params.shape[0]):
                        if torch.sum(torch.abs(gt_motion_params[i]-gt_motion_params[i-1])) > 0:
                            gt_motion_params_combined = torch.cat((gt_motion_params_combined, gt_motion_params[i:i+1,:]), dim=0)
                            gt_traj_combined[0].append(gt_traj[0][i]) 
                            gt_traj_combined[1].append(gt_traj[1][i])
                        else:
                            gt_traj_combined[0][-1] = np.concatenate((gt_traj_combined[0][-1], gt_traj[0][i]), axis=0)
                            gt_traj_combined[1][-1] = np.concatenate((gt_traj_combined[1][-1], gt_traj[1][i]), axis=0)

                    target_img_3D_coil = ifft2c_ndim(target_kspace_3D, 3)
                    input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, target_img_3D_coil, gt_motion_params_combined, gt_traj_combined, weight_rot=True, args=self.args,
                                                                        max_coil_size=self.args.train_nufft_max_coil_size)

                    if mode == 'motion_corrected':
                        # Correct motion corrupted undersampled k-space with gt motion parameters
                        input_corrected_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params_combined, gt_traj_combined, 
                                                                            weight_rot=True, args=self.args, do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                            max_coil_size=self.args.train_nufft_max_coil_size)
                        input_img_3D = complex_mul(input_corrected_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                    else:
                        if self.args.train_use_nufft_adjoint:
                            # Apply adjoint nufft with all zero motion parameters to obtain the corrupted coil images
                            input_corrupted_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, None, gt_traj_combined, weight_rot=True, args=self.args,
                                                                                do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                                max_coil_size=self.args.train_nufft_max_coil_size)
                            input_img_3D = complex_mul(input_corrupted_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                        else:
                            input_img_3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)


                # move corresponding axis to batch dimension
                input_img_2D = input_img_3D.moveaxis(ax_ind, 0)

                sens_maps_conj_3D = None
                sens_maps_3D = None
                input_kspace_3D = None
                target_kspace_3D = None
                input_corrupted_kspace3D = None
                target_img_3D_coil = None
                input_nufft_img3D_coil = None
                input_corrupted_img3D_coil = None
                input_corrected_img3D_coil = None
                input_img_3D = None

                recon_image_fg_1c = self.val_step(input_img_2D, binary_background_mask_2D)
                
                if mode == 'motion_corrupted':
                    # Implement alignment of reconstructions to target images
                    recon_image_fg_1c_aligned = align_3D_volumes(recon_image_fg_1c, target_image_fg_1c, binary_background_mask_2D, self.args.gpu)
                    psnr_tmp = torch.mean((recon_image_fg_1c_aligned - target_image_fg_1c)**2)
                    psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)

                    val_meters["PSNR_unet_aligned"].update(psnr_sense.item())

                psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)

                val_meters["PSNR_unet"].update(psnr_sense.item())

                # Compute PSNR for zero-filled recon
                input_img_2D_fg_1c = complex_abs(input_img_2D * binary_background_mask_2D)
                psnr_tmp = torch.mean((input_img_2D_fg_1c - target_image_fg_1c)**2)
                psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)
                
                val_meters["PSNR_zerofilled"].update(psnr_sense.item())

                if save_recon:
                    torch.save(recon_image_fg_1c, os.path.join(save_path_recon,f"recon_{filename[0].split('.')[0]}.pt"))

                    if mode == 'motion_corrupted':
                        torch.save(recon_image_fg_1c_aligned, os.path.join(save_path_recon,f"recon_aligned_{filename[0].split('.')[0]}.pt"))

                    list_of_slices = None
                    save_slice_images_from_volume(recon_image_fg_1c.cpu(), list_of_slices, save_path, f"recAxUnet_{filename[0].split('.')[0]}", axis_names = ["ax","cor","sag"], dir_name=f"slice_images")


            for name,meter in zip(val_meters.keys(), val_meters.values()):
                logging.info(f"Done evaluation for {save_name}! {name} avg: {meter.avg:.5f}, std: {meter.std:.5f}.")
            
            pickle.dump( val_meters, open(os.path.join(save_path, f"test_meter_{save_name}.pkl"), "wb" ) , pickle.HIGHEST_PROTOCOL )
        
    

    
        


            