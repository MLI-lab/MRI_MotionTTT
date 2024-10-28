
import torch
import numpy as np

from functions.utils.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim, fft2c_ndim, normalize_separate_over_ch
from functions.pre_training_src.train_base_module import TrainModuleBase

class UnetTrainModule(TrainModuleBase):
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

        target_kspace_hybrid_3D = ifft2c_ndim(target_kspace_3D.moveaxis(ax_ind+1, -2), 1).moveaxis(-2, 0)

        # select slices
        if self.args.train_on_motion_free_inputs:
            input_img_2D = input_img_3D[rec_id]
            target_kspace_2D = target_kspace_hybrid_3D[rec_id]
            sens_maps_2D = sens_maps_3D[rec_id]
            target_image_2D = target_img_3D[rec_id]
            binary_background_mask_2D = binary_background_mask_3D[rec_id]
        else:
            input_img_2D = None

        if input_cor_img3D is not None:
            input_cor_img_2D = input_cor_img3D[rec_id_cor]
            target_cor_kspace_2D = target_kspace_hybrid_3D[rec_id_cor]
            sens_maps_cor_2D = sens_maps_3D[rec_id_cor]
            target_cor_image_2D = target_img_3D[rec_id_cor]
            binary_background_mask_cor_2D = binary_background_mask_3D[rec_id_cor]

            # Stack along batch dimension
            if input_img_2D is not None:
                input_img_2D = torch.cat([input_img_2D, input_cor_img_2D], dim=0)
                target_image_2D = torch.cat([target_image_2D, target_cor_image_2D], dim=0)
                sens_maps_2D = torch.cat([sens_maps_2D, sens_maps_cor_2D], dim=0)
                target_kspace_2D = torch.cat([target_kspace_2D, target_cor_kspace_2D], dim=0)
                binary_background_mask_2D = torch.cat([binary_background_mask_2D, binary_background_mask_cor_2D], dim=0)
            else:
                input_img_2D = input_cor_img_2D
                target_image_2D = target_cor_image_2D
                sens_maps_2D = sens_maps_cor_2D
                target_kspace_2D = target_cor_kspace_2D
                binary_background_mask_2D = binary_background_mask_cor_2D


        if input_corMild_img3D is not None:
            input_corMild_img_2D = input_corMild_img3D[rec_id_corMild]
            target_corMild_kspace_2D = target_kspace_hybrid_3D[rec_id_corMild]
            sens_maps_corMild_2D = sens_maps_3D[rec_id_corMild]
            target_corMild_image_2D = target_img_3D[rec_id_corMild]
            binary_background_mask_corMild_2D = binary_background_mask_3D[rec_id_corMild]

            # Stack along batch dimension
            if input_img_2D is not None:
                input_img_2D = torch.cat([input_img_2D, input_corMild_img_2D], dim=0)
                target_image_2D = torch.cat([target_image_2D, target_corMild_image_2D], dim=0)
                sens_maps_2D = torch.cat([sens_maps_2D, sens_maps_corMild_2D], dim=0)
                target_kspace_2D = torch.cat([target_kspace_2D, target_corMild_kspace_2D], dim=0)
                binary_background_mask_2D = torch.cat([binary_background_mask_2D, binary_background_mask_corMild_2D], dim=0)
            else:
                input_img_2D = input_corMild_img_2D
                target_image_2D = target_corMild_image_2D
                sens_maps_2D = sens_maps_corMild_2D
                target_kspace_2D = target_corMild_kspace_2D
                binary_background_mask_2D = binary_background_mask_corMild_2D

        # Shuffle each tensor along the batch dimension using the same permutation
        perm = np.random.permutation(input_img_2D.shape[0])
        input_img_2D = input_img_2D[perm]
        target_image_2D = target_image_2D[perm]
        sens_maps_2D = sens_maps_2D[perm]
        target_kspace_2D = target_kspace_2D[perm]
        binary_background_mask_2D = binary_background_mask_2D[perm]

        assert self.args.batch_size <= input_img_2D.shape[0], "Batch size is larger than the number of slices"

        loss_tot = 0
        loss_img_tot = 0
        loss_ksp_tot = 0
        num_batches = input_img_2D.shape[0] // self.args.batch_size
        for i in range(num_batches):
            start = i * self.args.batch_size
            end = (i + 1) * self.args.batch_size

            model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(input_img_2D[start:end],-1,1), eps=1e-11)

            recon_image_full_2c = self.model(model_inputs_img_full_2c_norm)
            recon_image_full_2c = recon_image_full_2c * std + mean
            recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)                            # (batch, x, y, ch)

            if self.args.train_loss == "sup_ksp" or self.args.train_loss == "joint":
                recon_image_coil = complex_mul(recon_image_full_2c.unsqueeze(1), sens_maps_2D[start:end])          # (batch, coils, x, y, ch)
                recon_kspace_full = fft2c_ndim(recon_image_coil, 2)                                     # (batch, coils, x, y, ch)
            else:
                recon_kspace_full = None

            loss, loss_img, loss_ksp = self.train_loss_function(target_kspace_2D[start:end], recon_kspace_full, target_image_2D[start:end], recon_image_full_2c)

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=1.)
            self.optimizer.step()

            loss_tot += loss.item()
            loss_img_tot += loss_img.item()
            loss_ksp_tot += loss_ksp.item()

        loss_tot /= num_batches
        loss_img_tot /= num_batches
        loss_ksp_tot /= num_batches

        recon_image_fg_1c = complex_abs(recon_image_full_2c.detach() * binary_background_mask_2D[start:end])

        return recon_image_fg_1c, target_image_2D[start:end], input_img_2D[start:end], loss_tot, loss_img_tot, loss_ksp_tot
    
    def val_step(self, input_img_2D, binary_background_mask_2D):
        model_inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(input_img_2D,-1,1), eps=1e-11)
        recon_image_full_2c = self.model(model_inputs_img_full_2c_norm)
        recon_image_full_2c = recon_image_full_2c * std + mean
        recon_image_full_2c = torch.moveaxis(recon_image_full_2c, 1, -1)                    # (batch, x, y, ch)

        recon_image_fg_1c = complex_abs(recon_image_full_2c * binary_background_mask_2D)
        
        return recon_image_fg_1c 
