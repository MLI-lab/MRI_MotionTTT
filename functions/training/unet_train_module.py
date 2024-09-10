import torch
import os
import pickle
import logging
import numpy as np
import argparse

from functions.helpers.meters import AverageMeter, TrackMeter
from functions.helpers.progress_bar import ProgressBar
from functions.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim, fft2c_ndim, normalize_separate_over_ch
from functions.helpers.helpers_log_save_image_utils import add_img_to_tensorboard, save_figure
from functions.training.losses import SSIMLoss

from functions.motion_simulation.motion_functions import motion_correction_NUFFT#, generate_random_motion_params
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT


def init_train_val_meters(args):

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
    
    if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint:
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
    else:
        val_meters_per_epoch = {
            'SSIM' : AverageMeter(), 
            'PSNR' : AverageMeter(),    
            'L1_mag' : AverageMeter(), 
            'L1_comp' : AverageMeter(), 
            'L1_ksp' : AverageMeter(),
            }
        val_meters_over_epochs = {
            'SSIM' : TrackMeter('increasing'), 
            'PSNR' : TrackMeter('increasing'),
            'L1_mag' : TrackMeter('decaying'),
            'L1_comp' : TrackMeter('decaying'),
            'L1_ksp' : TrackMeter('decaying'),
            } 




    return train_meters_per_epoch, train_meters_over_epochs, val_meters_per_epoch, val_meters_over_epochs


class UnetTrainModule():

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

        self.train_meters_per_epoch, self.train_meters_over_epochs, self.val_meters_per_epoch, self.val_meters_over_epochs = init_train_val_meters(args)

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)

        self.psnrs = []
        self.val_max_mot = None
        self.val_Ns = None
        self.val_random_motion_seed = None

    def train_epoch_volume(self, epoch):
        self.model.train()
        train_bar = ProgressBar(self.train_loader, epoch)
        for meter in self.train_meters_per_epoch.values():
            meter.reset()
        
        for batch_id, batch in enumerate(train_bar):

            target_kspace_3D, input_kspace_3D, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename = batch

            # assume batch size is 1
            target_kspace_3D = target_kspace_3D[0].cuda(self.args.gpu)                                       # (coils, x, y, z, ch)
            input_kspace_3D = input_kspace_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
            binary_background_mask_3D = binary_background_mask_3D[0].cuda(self.args.gpu)                     # (x, y, z, 1)
            sens_maps_3D = sens_maps_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
            sens_maps_conj_3D = sens_maps_conj_3D[0].cuda(self.args.gpu)                                     # (coils, x, y, z, ch)
            target_img_3D = target_img_3D[0].cuda(self.args.gpu)                                             # (x, y, z, ch)
            mask3D = mask3D[0].cuda(self.args.gpu)                                                           # (1, x, y, z, 1)

            if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                assert isinstance(self.args.train_Ns, list) and isinstance(self.args.train_max_rots, list) and isinstance(self.args.train_max_trans, list)
                if self.args.train_on_motion_corrected_inputs and self.args.train_on_motion_corrupted_inputs:
                    raise ValueError("Can only train on either motion corrected or corrupted inputs, not both.")
                
                Ns = np.random.choice(self.args.train_Ns)
                max_rot = np.random.choice(self.args.train_max_rots)
                max_trans = max_rot #np.random.choice(self.args.train_max_trans)
                random_motion_seed = np.random.randint(100, self.args.train_num_random_motion_seeds+100)

                traj,_ = generate_interleaved_cartesian_trajectory(Ns, mask3D, self.args)
                gt_motion_params = generate_random_motion_params(Ns-1, max_trans, max_rot, random_motion_seed).cuda(self.args.gpu)

                input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, gt_motion_params, traj, weight_rot=True, args=self.args)

                if self.args.train_on_motion_corrupted_inputs:
                    input_cor_img3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3) , sens_maps_conj_3D).sum(dim=0, keepdim=False)

                elif self.args.train_on_motion_corrected_inputs:
                    input_corrected_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params, traj, weight_rot=True, args=self.args,
                                                                         do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                    input_cor_img3D = complex_mul(input_corrected_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)

            if self.args.train_use_nufft_adjoint:
                traj,_ = generate_interleaved_cartesian_trajectory(self.args.Ns, mask3D, self.args)

                input_nufft_img3D_coil = motion_correction_NUFFT(input_kspace_3D, None, traj, weight_rot=True, args=self.args,
                                                                            do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                input_img_3D = complex_mul(input_nufft_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
            else:
                input_img_3D = complex_mul(ifft2c_ndim(input_kspace_3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)

            for ax_ind, batch_size in enumerate(self.args.train_batch_size_per_axis):
                if batch_size:
                    # move corresponding axis to batch dimension
                    input_img_3D = input_img_3D.moveaxis(ax_ind, 0)
                    sens_maps_3D = sens_maps_3D.moveaxis(ax_ind+1, 0)
                    target_img_3D = target_img_3D.moveaxis(ax_ind, 0)
                    binary_background_mask_3D = binary_background_mask_3D.moveaxis(ax_ind, 0)
                    target_kspace_hybrid_3D = ifft2c_ndim(target_kspace_3D.moveaxis(ax_ind+1, -2), 1).moveaxis(-2, 0)

                    # select random slice indices
                    rec_id = np.random.choice(range(input_img_3D.shape[0]),size=(batch_size), replace=False)

                    # select slices
                    input_img_2D = input_img_3D[rec_id]
                    target_kspace_2D = target_kspace_hybrid_3D[rec_id]
                    sens_maps_2D = sens_maps_3D[rec_id]
                    target_image_2D = target_img_3D[rec_id]
                    binary_background_mask_2D = binary_background_mask_3D[rec_id]

                    if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                        input_cor_img3D = input_cor_img3D.moveaxis(ax_ind, 0)
                        input_cor_img_full_2D = input_cor_img3D[rec_id]

                    if self.args.train_one_grad_step_per_image_in_batch:
                        loss = 0
                        loss_img = 0
                        loss_ksp = 0
                        for i in range(batch_size):
                            recon_image_full_2c, loss_tmp, loss_img_tmp, loss_ksp_tmp = self.train_step(input_img_2D[i:i+1], sens_maps_2D[i:i+1], target_image_2D[i:i+1], target_kspace_2D[i:i+1])
                            loss += loss_tmp
                            loss_img += loss_img_tmp
                            loss_ksp += loss_ksp_tmp
                        loss /= batch_size
                        loss_img /= batch_size
                        loss_ksp /= batch_size
                    else:
                        recon_image_full_2c, loss, loss_img, loss_ksp = self.train_step(input_img_2D, sens_maps_2D, target_image_2D, target_kspace_2D)

                    name_tag = f"_ax{ax_ind}"
                    self.train_step_logging(epoch, batch_id, input_img_2D[0:1], target_image_2D[0:1], recon_image_full_2c[0:1], loss, loss_img, loss_ksp, binary_background_mask_2D[0:1], name_tag)
                    input_img_3D = input_img_3D.moveaxis(0, ax_ind)
                    sens_maps_3D = sens_maps_3D.moveaxis(0, ax_ind+1)
                    target_img_3D = target_img_3D.moveaxis(0, ax_ind)
                    binary_background_mask_3D = binary_background_mask_3D.moveaxis(0, ax_ind)

                    if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                        recon_image_full_2c, loss, loss_img, loss_ksp = self.train_step(input_cor_img_full_2D, sens_maps_2D, target_image_2D, target_kspace_2D)
                        name_tag = f"_ax{ax_ind}_cor"
                        self.train_step_logging(epoch, batch_id, input_cor_img_full_2D[0:1], target_image_2D[0:1], recon_image_full_2c[0:1], loss, loss_img, loss_ksp, binary_background_mask_2D[0:1], name_tag)
                        input_cor_img3D = input_cor_img3D.moveaxis(0, ax_ind)

        self.train_epoch_logging(epoch)
        if self.scheduler:
            self.scheduler.step()

    def train_step(self, inputs_img_full_2c, sens_maps, target_image_full_2c, target_kspace):
        model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(inputs_img_full_2c,-1,1), eps=1e-11)
        #print(model_inputs_img_full_2c_norm.shape)

        recon_image_full_2c = self.model(model_inputs_img_full_2c_norm)
        recon_image_full_2c = recon_image_full_2c * std + mean
        recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)                            # (batch, x, y, ch)

        if self.args.train_loss == "sup_ksp" or self.args.train_loss == "joint":
            recon_image_coil = complex_mul(recon_image_full_2c.unsqueeze(1), sens_maps)             # (batch, coils, x, y, ch)
            recon_kspace_full = fft2c_ndim(recon_image_coil, 2)                                     # (batch, coils, x, y, ch)
        else:
            recon_kspace_full = None

        loss, loss_img, loss_ksp = self.train_loss_function(target_kspace, recon_kspace_full, target_image_full_2c, recon_image_full_2c)

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=1.)
        self.optimizer.step()

        return recon_image_full_2c, loss.item(), loss_img.item(), loss_ksp.item()     



    def train_epoch(self, epoch):
        self.model.train()
        train_bar = ProgressBar(self.train_loader, epoch)
        for meter in self.train_meters_per_epoch.values():
            meter.reset()
        
        for batch_id, batch in enumerate(train_bar):

            inputs_img_full_2c, target_image_full_2c, target_kspace, sens_maps, sens_maps_conj, binary_background_mask, filename, slice_num, axis = batch

            if not self.args.load_data_to_gpu:
                inputs_img_full_2c = inputs_img_full_2c.cuda(self.args.gpu)                             # (batch, x, y, ch)
                target_kspace = target_kspace.cuda(self.args.gpu)                                       # (batch, coils, x, y, ch)
                target_image_full_2c = target_image_full_2c.cuda(self.args.gpu)                         # (batch, x, y, ch)
                sens_maps = sens_maps.cuda(self.args.gpu)                                               # (batch, coils, x, y, ch)
                sens_maps_conj = sens_maps_conj.cuda(self.args.gpu)                                     # (batch, coils, x, y, ch)
                binary_background_mask = binary_background_mask.cuda(self.args.gpu)                     # (batch, x, y, 1)


            recon_image_full_2c, loss, loss_img, loss_ksp = self.train_step(inputs_img_full_2c, sens_maps, target_image_full_2c, target_kspace)

            self.train_step_logging(epoch, batch_id, inputs_img_full_2c, target_image_full_2c, recon_image_full_2c, loss, loss_img, loss_ksp, binary_background_mask)

        self.train_epoch_logging(epoch)

        if self.scheduler:
            self.scheduler.step()

    def train_step_logging(self, epoch, batch_id, inputs_img_full_2c, target_image_full_2c, recon_image_full_2c, loss, loss_img, loss_ksp, binary_background_mask, name_tag=""):

        self.train_meters_per_epoch["train_loss"].update(loss)
        self.train_meters_per_epoch["train_loss_img"].update(loss_img)
        self.train_meters_per_epoch["train_loss_ksp"].update(loss_ksp)

        if epoch % self.args.log_imgs_to_tb_every == 0 and batch_id in [0] and self.tb_writer:
            recon_image_fg_2c = recon_image_full_2c.detach() * binary_background_mask
            target_image_fg_2c = target_image_full_2c * binary_background_mask
            add_img_to_tensorboard(self.tb_writer, epoch, f"train_vis_{batch_id}"+name_tag, inputs_img_full_2c, target_image_fg_2c, recon_image_fg_2c, self.ssim_loss)

    def train_epoch_logging(self, epoch):
        self.train_meters_over_epochs["train_loss"].update(self.train_meters_per_epoch["train_loss"].avg, epoch)
        self.train_meters_over_epochs["train_loss_img"].update(self.train_meters_per_epoch["train_loss_img"].avg, epoch)
        self.train_meters_over_epochs["train_loss_ksp"].update(self.train_meters_per_epoch["train_loss_ksp"].avg, epoch)

        if self.tb_writer:
            self.tb_writer.add_scalar("Train loss total", self.train_meters_per_epoch["train_loss"].avg, epoch)
            self.tb_writer.add_scalar("Train loss img", self.train_meters_per_epoch["train_loss_img"].avg, epoch)
            self.tb_writer.add_scalar("Train loss ksp", self.train_meters_per_epoch["train_loss_ksp"].avg, epoch)

    def val_epoch_volume(self, epoch):
        self.model.eval()
        val_bar = ProgressBar(self.val_loader, epoch)
        for meter in self.val_meters_per_epoch.values():
            meter.reset()

        with torch.no_grad():
            self.psnrs = []
            for batch_id, batch in enumerate(val_bar):

                target_kspace_3D, input_kspace_3D, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename = batch

                # assume batch size is 1
                target_kspace_3D = target_kspace_3D[0].cuda(self.args.gpu)                                       # (coils, x, y, z, ch)
                input_kspace_3D = input_kspace_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
                binary_background_mask_3D = binary_background_mask_3D[0].cuda(self.args.gpu)                     # (x, y, z, 1)
                sens_maps_3D = sens_maps_3D[0].cuda(self.args.gpu)                                               # (coils, x, y, z, ch)
                sens_maps_conj_3D = sens_maps_conj_3D[0].cuda(self.args.gpu)                                     # (coils, x, y, z, ch)
                target_img_3D = target_img_3D[0].cuda(self.args.gpu)                                             # (x, y, z, ch)
                mask3D = mask3D[0].cuda(self.args.gpu)                                                           # (1, x, y, z, 1)

                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    assert isinstance(self.args.train_Ns, list) and isinstance(self.args.train_max_rots, list) and isinstance(self.args.train_max_trans, list)
                    if self.args.train_on_motion_corrected_inputs and self.args.train_on_motion_corrupted_inputs:
                        raise ValueError("Can only train on either motion corrected or corrupted inputs, not both.")
                    
                    Ns = np.random.choice(self.args.train_Ns)
                    max_rot = np.random.choice(self.args.train_max_rots)
                    max_trans = max_rot #np.random.choice(self.args.train_max_trans)
                    random_motion_seed = np.random.randint(100, self.args.train_num_random_motion_seeds+100)
                    self.val_max_mot = max_rot
                    self.val_Ns = Ns
                    self.val_random_motion_seed = random_motion_seed

                    traj,_ = generate_interleaved_cartesian_trajectory(Ns, mask3D, self.args)
                    gt_motion_params = generate_random_motion_params(Ns-1, max_trans, max_rot, random_motion_seed).cuda(self.args.gpu)

                    input_corrupted_kspace3D = motion_corruption_NUFFT(target_kspace_3D, gt_motion_params, traj, weight_rot=True, args=self.args)

                    if self.args.train_on_motion_corrupted_inputs:
                        input_cor_img3D = complex_mul(ifft2c_ndim(input_corrupted_kspace3D, 3) , sens_maps_conj_3D).sum(dim=0, keepdim=False)

                    elif self.args.train_on_motion_corrected_inputs:
                        input_corrected_img3D_coil = motion_correction_NUFFT(input_corrupted_kspace3D, -1* gt_motion_params, traj, weight_rot=True, args=self.args,
                                                                            do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                        input_cor_img3D = complex_mul(input_corrected_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)

                if self.args.train_use_nufft_adjoint:
                    traj,_ = generate_interleaved_cartesian_trajectory(self.args.Ns, mask3D, self.args)

                    input_nufft_img3D_coil = motion_correction_NUFFT(input_kspace_3D, None, traj, weight_rot=True, args=self.args,
                                                                                do_dcomp=self.args.train_use_nufft_with_dcomp, num_iters_dcomp=3)
                    input_img_3D = complex_mul(input_nufft_img3D_coil, sens_maps_conj_3D).sum(dim=0, keepdim=False)
                else:
                    input_img_3D = complex_mul(ifft2c_ndim(input_kspace_3D, 3), sens_maps_conj_3D).sum(dim=0, keepdim=False)  


                # move corresponding axis to batch dimension
                ax_ind = 2
                input_img_2D = input_img_3D.moveaxis(ax_ind, 0)
                #sens_maps_2D = sens_maps_3D.moveaxis(ax_ind+1, 0)
                target_image_2D = target_img_3D.moveaxis(ax_ind, 0)
                binary_background_mask_2D = binary_background_mask_3D.moveaxis(ax_ind, 0)
                
                target_image_fg_2c = target_image_2D * binary_background_mask_2D                          # (batch, x, y, ch)
                target_image_fg_1c = complex_abs(target_image_fg_2c).unsqueeze(1)                           # (batch, 1, x, y)

                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    input_cor_img_full_2D = input_cor_img3D.moveaxis(ax_ind, 0)

                # pass motion free data through unet
                model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(input_img_2D,-1,1), eps=1e-11)
                recon_image_full_2c = self.model(model_inputs_img_full_2c_norm)
                recon_image_full_2c = recon_image_full_2c * std + mean
                recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)   

                recon_image_fg_2c = recon_image_full_2c * binary_background_mask_2D                            # (batch, x, y, ch)
                recon_image_fg_1c = complex_abs(recon_image_fg_2c).unsqueeze(1)                             # (batch, 1, x, y)

                psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)
                self.val_meters_per_epoch["PSNR"].update(psnr_sense.item())
                self.psnrs.append(psnr_sense.item())

                if (epoch % self.args.log_imgs_to_tb_every == 0 or epoch in [0,1,2,3,4] ) and batch_id in [0,1] and self.tb_writer:
                    slice_num = input_img_2D.shape[0]//2
                    add_img_to_tensorboard(self.tb_writer, epoch, f"val_motfree_{filename[0].split('.')[0]}_ax{ax_ind}_s{slice_num}", input_img_2D[slice_num:slice_num+1], target_image_fg_2c[slice_num:slice_num+1], recon_image_fg_2c[slice_num:slice_num+1], self.ssim_loss)

                    slice_num = input_img_2D.shape[2]//2
                    add_img_to_tensorboard(self.tb_writer, epoch, f"val_motfree_{filename[0].split('.')[0]}_ax1_s{slice_num}", input_img_2D[:,:,slice_num].unsqueeze(0), target_image_fg_2c[:,:,slice_num].unsqueeze(0), recon_image_fg_2c[:,:,slice_num].unsqueeze(0), self.ssim_loss)
            

                if self.args.train_on_motion_corrected_inputs or self.args.train_on_motion_corrupted_inputs:
                    # pass motion corrupted/corrected data through unet
                    model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(input_cor_img_full_2D,-1,1), eps=1e-11)
                    recon_image_full_2c = self.model(model_inputs_img_full_2c_norm)
                    recon_image_full_2c = recon_image_full_2c * std + mean
                    recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)   

                    recon_image_fg_2c = recon_image_full_2c * binary_background_mask_2D                            # (batch, x, y, ch)
                    recon_image_fg_1c = complex_abs(recon_image_fg_2c).unsqueeze(1)                             # (batch, 1, x, y)

                    psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                    psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)
                    if self.args.train_on_motion_corrected_inputs:
                        self.val_meters_per_epoch["PSNR_corrected"].update(psnr_sense.item())
                    elif self.args.train_on_motion_corrupted_inputs:
                        self.val_meters_per_epoch["PSNR_corrupted"].update(psnr_sense.item())

                    if (epoch % self.args.log_imgs_to_tb_every == 0 or epoch in [0,1,2,3,4] ) and batch_id in [0,1] and self.tb_writer:
                        slice_num = input_img_2D.shape[0]//2
                        add_img_to_tensorboard(self.tb_writer, epoch, f"val_cor_{filename[0].split('.')[0]}_ax{ax_ind}_s{slice_num}", input_cor_img_full_2D[slice_num:slice_num+1], target_image_fg_2c[slice_num:slice_num+1], recon_image_fg_2c[slice_num:slice_num+1], self.ssim_loss)

                        slice_num = input_img_2D.shape[2]//2
                        add_img_to_tensorboard(self.tb_writer, epoch, f"val_cor_{filename[0].split('.')[0]}_ax1_s{slice_num}", input_img_2D[:,:,slice_num].unsqueeze(0), target_image_fg_2c[:,:,slice_num].unsqueeze(0), recon_image_fg_2c[:,:,slice_num].unsqueeze(0), self.ssim_loss)
            
                
            for name,meter in zip(self.val_meters_per_epoch.keys(), self.val_meters_per_epoch.values()):
                self.val_meters_over_epochs[name].update(meter.avg, epoch)
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"Val {name}", meter.avg, epoch)      
          

    def val_epoch(self, epoch):
        self.model.eval()
        val_bar = ProgressBar(self.val_loader, epoch)
        for meter in self.val_meters_per_epoch.values():
            meter.reset()

        with torch.no_grad():
            for batch_id, batch in enumerate(val_bar):

                inputs_img_full_2c, target_image_full_2c, target_kspace, sens_maps, sens_maps_conj, binary_background_mask, filename, slice_num, axis = batch

                if not self.args.load_data_to_gpu:
                    inputs_img_full_2c = inputs_img_full_2c.cuda(self.args.gpu)                             # (batch, x, y, ch)
                    target_kspace = target_kspace.cuda(self.args.gpu)                                       # (batch, coils, x, y, ch)
                    target_image_full_2c = target_image_full_2c.cuda(self.args.gpu)                         # (batch, x, y, ch)
                    sens_maps = sens_maps.cuda(self.args.gpu)                                               # (batch, coils, x, y, ch)
                    sens_maps_conj = sens_maps_conj.cuda(self.args.gpu)                                     # (batch, coils, x, y, ch)
                    binary_background_mask = binary_background_mask.cuda(self.args.gpu)                     # (batch, x, y, 1)


                inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(inputs_img_full_2c,-1,1), eps=1e-11)

                # Compute reconstructions
                recon_image_full_2c = self.model(inputs_img_full_2c_norm)
                recon_image_full_2c = recon_image_full_2c * std + mean
                recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)                            # (batch, x, y, ch)

                recon_image_fg_2c = recon_image_full_2c * binary_background_mask                            # (batch, x, y, ch)
                recon_image_fg_1c = complex_abs(recon_image_fg_2c).unsqueeze(1)                             # (batch, 1, x, y)

                recon_image_coil = complex_mul(recon_image_full_2c.unsqueeze(1), sens_maps)                 # (batch, coils, x, y, ch)
                recon_kspace_full = fft2c_ndim(recon_image_coil, 2)                                         # (batch, coils, x, y, ch)


                # These target images are either rss or sense depending on the dataloader
                target_image_fg_2c = target_image_full_2c * binary_background_mask                          # (batch, x, y, ch)
                target_image_fg_1c = complex_abs(target_image_fg_2c).unsqueeze(1)                           # (batch, 1, x, y)

                # Compute val scores
                L1_comp = torch.sum(torch.abs(recon_image_fg_2c - target_image_fg_2c)) / torch.sum(torch.abs(target_image_fg_2c))
                self.val_meters_per_epoch["L1_comp"].update(L1_comp.item())

                L1_mag = torch.sum(torch.abs(recon_image_fg_1c - target_image_fg_1c)) / torch.sum(torch.abs(target_image_fg_1c))
                self.val_meters_per_epoch["L1_mag"].update(L1_mag.item())

                L1_ksp = torch.sum(torch.abs(recon_kspace_full - target_kspace)) / torch.sum(torch.abs(target_kspace))
                self.val_meters_per_epoch["L1_ksp"].update(L1_ksp.item())

                ssim = 1-self.ssim_loss(recon_image_fg_1c, target_image_fg_1c, data_range=target_image_fg_1c.max().unsqueeze(0))
                self.val_meters_per_epoch["SSIM"].update(ssim.item())

                psnr_tmp = torch.mean((recon_image_fg_1c - target_image_fg_1c)**2)
                psnr_sense = 20 * torch.log10(torch.tensor(target_image_fg_1c.max().item()))- 10 * torch.log10(psnr_tmp)
                self.val_meters_per_epoch["PSNR"].update(psnr_sense.item())

                if epoch % self.args.log_imgs_to_tb_every == 0 and batch_id in [0,1] and self.tb_writer:
                    add_img_to_tensorboard(self.tb_writer, epoch, f"val_{filename[0].split('.')[0]}_{axis[0]}_s{slice_num[0]}", inputs_img_full_2c, target_image_fg_2c, recon_image_fg_2c, self.ssim_loss)
            
            for name,meter in zip(self.val_meters_per_epoch.keys(), self.val_meters_per_epoch.values()):
                self.val_meters_over_epochs[name].update(meter.avg, epoch)
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"Val {name}", meter.avg, epoch)

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
            text_to_log += f"PSNRs per val vol (mot free): {self.psnrs} | Max mot: {self.val_max_mot} | Ns: {self.val_Ns} | Mot Seed: {self.val_random_motion_seed}"
        logging.info(text_to_log)

    def save_checkpoints(self, epoch, save_metrics):
        torch.save((self.model.state_dict()), os.path.join(self.args.checkpoint_path, "checkpoint_last.pt"))

        for save_metric in save_metrics:
            if self.val_meters_over_epochs[save_metric].best_epoch == epoch:
                #torch.save((self.model.state_dict()), os.path.join(self.args.checkpoint_path , f"checkpoint_best_{save_metric}.pt"))

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
        