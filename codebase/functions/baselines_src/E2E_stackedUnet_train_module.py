

import torch
import numpy as np
import math

from functions.utils.helpers.helpers_math import complex_abs, normalize_separate_over_ch
from functions.pre_training_src.train_base_module import TrainModuleBase


class StackedUnetTrainModule(TrainModuleBase):
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

        super().__init__(args, train_loader, val_loader, model, optimizer, scheduler, train_loss_function, tb_writer)


    def train_step(self, input_img_3D, input_cor_img3D, input_corMild_img3D, sens_maps_3D, target_img_3D, target_kspace_3D, binary_background_mask_3D, rec_id, rec_id_cor, rec_id_corMild, ax_ind):
        target_kspace_2D = None
        sens_maps_2D = None

        data_range = torch.max(complex_abs(target_img_3D)).cuda(self.args.gpu).repeat(self.args.batch_size)

        # select slices
        if self.args.train_on_motion_free_inputs:
            input_img_2D = complex_abs(input_img_3D[rec_id]).unsqueeze(1)    
            target_image_2D = complex_abs(target_img_3D[rec_id]).unsqueeze(1)
            binary_background_mask_2D = binary_background_mask_3D[rec_id]
        
            rec_id_next = rec_id + 1
            if np.max(rec_id_next) == input_img_3D.shape[0]:
                rec_id_next[rec_id_next == input_img_3D.shape[0]] = input_img_3D.shape[0] - 1

            rec_id_prev = rec_id - 1
            if np.min(rec_id_prev) == -1:
                rec_id_prev[rec_id_prev == -1] = 0

            input_img_2D_next = complex_abs(input_img_3D[rec_id_next]).unsqueeze(1)
            input_img_2D_prev = complex_abs(input_img_3D[rec_id_prev]).unsqueeze(1)
        else:
            input_img_2D = None

        if input_cor_img3D is not None:
            rec_id_cor_next = rec_id_cor + 1
            if np.max(rec_id_cor_next) == input_cor_img3D.shape[0]:
                rec_id_cor_next[rec_id_cor_next == input_cor_img3D.shape[0]] = input_cor_img3D.shape[0] - 1

            rec_id_cor_prev = rec_id_cor - 1
            if np.min(rec_id_cor_prev) == -1:
                rec_id_cor_prev[rec_id_cor_prev == -1] = 0
            
            input_cor_img_2D = complex_abs(input_cor_img3D[rec_id_cor]).unsqueeze(1)
            input_cor_img_2D_next = complex_abs(input_cor_img3D[rec_id_cor_next]).unsqueeze(1)
            input_cor_img_2D_prev = complex_abs(input_cor_img3D[rec_id_cor_prev]).unsqueeze(1)

            target_cor_image_2D = complex_abs(target_img_3D[rec_id_cor]).unsqueeze(1)
            binary_background_mask_cor_2D = binary_background_mask_3D[rec_id_cor]

            # Stack along batch dimension
            if input_img_2D is not None:
                input_img_2D = torch.cat([input_img_2D, input_cor_img_2D], dim=0)
                input_img_2D_next = torch.cat([input_img_2D_next, input_cor_img_2D_next], dim=0)
                input_img_2D_prev = torch.cat([input_img_2D_prev, input_cor_img_2D_prev], dim=0)
                target_image_2D = torch.cat([target_image_2D, target_cor_image_2D], dim=0)
                binary_background_mask_2D = torch.cat([binary_background_mask_2D, binary_background_mask_cor_2D], dim=0)
            else:
                input_img_2D = input_cor_img_2D
                input_img_2D_next = input_cor_img_2D_next
                input_img_2D_prev = input_cor_img_2D_prev
                target_image_2D = target_cor_image_2D
                binary_background_mask_2D = binary_background_mask_cor_2D

        if input_corMild_img3D is not None:
            rec_id_corMild_next = rec_id_corMild + 1
            if np.max(rec_id_corMild_next) == input_corMild_img3D.shape[0]:
                rec_id_corMild_next[rec_id_corMild_next == input_corMild_img3D.shape[0]] = input_corMild_img3D.shape[0] - 1

            rec_id_corMild_prev = rec_id_corMild - 1
            if np.min(rec_id_corMild_prev) == -1:
                rec_id_corMild_prev[rec_id_corMild_prev == -1] = 0

            input_corMild_img_2D = complex_abs(input_corMild_img3D[rec_id_corMild]).unsqueeze(1)
            input_corMild_img_2D_next = complex_abs(input_corMild_img3D[rec_id_corMild_next]).unsqueeze(1)
            input_corMild_img_2D_prev = complex_abs(input_corMild_img3D[rec_id_corMild_prev]).unsqueeze(1)

            target_corMild_image_2D = complex_abs(target_img_3D[rec_id_corMild]).unsqueeze(1)
            binary_background_mask_corMild_2D = binary_background_mask_3D[rec_id_corMild]

            # Stack along batch dimension
            if input_img_2D is not None:
                input_img_2D = torch.cat([input_img_2D, input_corMild_img_2D], dim=0)
                input_img_2D_next = torch.cat([input_img_2D_next, input_corMild_img_2D_next], dim=0)
                input_img_2D_prev = torch.cat([input_img_2D_prev, input_corMild_img_2D_prev], dim=0)
                target_image_2D = torch.cat([target_image_2D, target_corMild_image_2D], dim=0)
                binary_background_mask_2D = torch.cat([binary_background_mask_2D, binary_background_mask_corMild_2D], dim=0)
            else:
                input_img_2D = input_corMild_img_2D
                input_img_2D_next = input_corMild_img_2D_next
                input_img_2D_prev = input_corMild_img_2D_prev
                target_image_2D = target_corMild_image_2D
                binary_background_mask_2D = binary_background_mask_corMild_2D

        # Shuffle each tensor along the batch dimension using the same permutation
        perm = np.random.permutation(input_img_2D.shape[0])
        input_img_2D = input_img_2D[perm]
        input_img_2D_next = input_img_2D_next[perm]
        input_img_2D_prev = input_img_2D_prev[perm]
        target_image_2D = target_image_2D[perm]
        binary_background_mask_2D = binary_background_mask_2D[perm]

        assert self.args.batch_size <= input_img_2D.shape[0], "Batch size is larger than the number of slices"

        loss_tot = 0
        loss_img_tot = 0
        loss_ksp_tot = 0
        num_batches = input_img_2D.shape[0] // self.args.batch_size
        counter = 0
        for i in range(num_batches):
            start = i * self.args.batch_size
            end = (i + 1) * self.args.batch_size

            if torch.sum(torch.abs(target_image_2D[start:end])) == 0:
                continue

            model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(input_img_2D[start:end], eps=1e-11)
            model_inputs_img_full_2c_norm_next, _, _ = normalize_separate_over_ch(input_img_2D_next[start:end], eps=1e-11)
            model_inputs_img_full_2c_norm_prev, _, _ = normalize_separate_over_ch(input_img_2D_prev[start:end], eps=1e-11)

            recon_image_full_1c = self.model(model_inputs_img_full_2c_norm_prev, model_inputs_img_full_2c_norm, model_inputs_img_full_2c_norm_next)
            recon_image_full_1c = recon_image_full_1c * std + mean                            # (batch, ch, x, y), required for ssim loss
            

            recon_kspace_full = None
            if self.args.train_loss == "ssim":
                loss_img = self.ssim_loss(recon_image_full_1c, target_image_2D[start:end], data_range=data_range)
                loss_ksp = torch.tensor(0.0)
                loss = loss_img
            else:
                loss, loss_img, loss_ksp = self.train_loss_function(target_kspace_2D, recon_kspace_full, target_image_2D[start:end], recon_image_full_1c)

            if math.isnan(loss.item()):
                print("Nan loss encountered.")
                

            #loss_img = torch.sum(torch.abs(recon_image_full_1c - target_image_2D)) / torch.sum(torch.abs(target_image_2D))
            #loss_ksp = torch.tensor(0.0)
            #loss = loss_img + loss_ksp

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=1.)
            self.optimizer.step()

            loss_tot += loss.item()
            #print(loss.item())
            loss_img_tot += loss_img.item()
            loss_ksp_tot += loss_ksp.item()
            counter += 1

        loss_tot /= counter
        loss_img_tot /= counter
        loss_ksp_tot /= counter

        recon_image_full_1c = torch.moveaxis(recon_image_full_1c, 1, -1)                            # (batch, x, y, ch)
        recon_image_fg_1c = recon_image_full_1c.detach() * binary_background_mask_2D[start:end]

        return recon_image_fg_1c[:,:,:,0], target_image_2D[start:end], input_img_2D[start:end], loss_tot, loss_img_tot, loss_ksp_tot
    

    def val_step(self, input_img_2D, binary_background_mask_2D, batch_size=None):
        input_img_2D = complex_abs(input_img_2D).unsqueeze(1)
        input_img_2D_next = input_img_2D[1:]
        input_img_2D_next = torch.cat([input_img_2D_next, input_img_2D_next[-1].unsqueeze(0)], dim=0)
        input_img_2D_prev = input_img_2D[:-1]
        input_img_2D_prev = torch.cat([input_img_2D[0].unsqueeze(0), input_img_2D_prev], dim=0)

        model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(input_img_2D, eps=1e-11)
        model_inputs_img_full_2c_norm_next, _, _ = normalize_separate_over_ch(input_img_2D_next, eps=1e-11)
        model_inputs_img_full_2c_norm_prev, _, _ = normalize_separate_over_ch(input_img_2D_prev, eps=1e-11)
        del input_img_2D, input_img_2D_next, input_img_2D_prev


        if batch_size:        
            # pass inputs through the model in batches
            recon_image_full_1c = torch.zeros_like(model_inputs_img_full_2c_norm)
            num_batches = int(np.ceil(model_inputs_img_full_2c_norm.shape[0] / batch_size))


            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, model_inputs_img_full_2c_norm.shape[0])
                recon_image_full_1c[start:end] = self.model(model_inputs_img_full_2c_norm_prev[start:end], model_inputs_img_full_2c_norm[start:end], model_inputs_img_full_2c_norm_next[start:end])
        else:
            recon_image_full_1c = self.model(model_inputs_img_full_2c_norm_prev, model_inputs_img_full_2c_norm, model_inputs_img_full_2c_norm_next)
        recon_image_full_1c = recon_image_full_1c * std + mean
        recon_image_full_1c = torch.moveaxis(recon_image_full_1c, 1, -1)                    # (batch, x, y, ch)

        recon_image_fg_1c = recon_image_full_1c * binary_background_mask_2D

        return recon_image_fg_1c[:,:,:,0]