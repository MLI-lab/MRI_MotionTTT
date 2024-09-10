
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

from functions.training.losses import SSIMLoss
from functions.helpers.meters import AverageMeter, TrackMeter, TrackMeter_testing
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj, normalize_separate_over_ch

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import motion_correction_NUFFT, unet_forward_all_axes
from functions.motion_simulation.motion_functions import motion_corruption_NUFFT

from functions.test_time_training.unet_TTT_base_module import UnetTTTModuleBase

from functions.data.data_loaders import invivo_loader


class UnetTTTModuleVivo(UnetTTTModuleBase):

    def __init__(
            self,
            args,
            model,
            ) -> None:
        
        super().__init__(args, model)

        self.TTT_meters_per_example = self.init_TTT_meters()

    def init_TTT_meters(self):
        # !!!!
        #  Decide what quantities to track during TTT

        TTT_meters_per_example = {
            "TTT_loss" : TrackMeter('decaying'),
            "ax_inds" : TrackMeter('decaying'),
        } 
            
        return TTT_meters_per_example

    def load_data_init_motion(self, evaluate_before_TTT):

        masked_corrupted_kspace3D, masked_motion_free_kspace3D, mask3D, smaps3D_conj, traj, binary_background_mask, smaps3D = invivo_loader(self.args, self.args.TTT_results_path)
        
        self.gt_motion_params = None
        self.phase = 0

        if self.args.TTT_path_to_pred_motion_params is not None:
            pred_motion_params_over_epochs = torch.from_numpy(np.load(self.args.TTT_path_to_pred_motion_params))
            self.loaded_pred_motion_params = pred_motion_params_over_epochs[:,:,-1]
            with open(self.args.TTT_path_to_pred_motion_params,'rb') as fn:
                final_results_dict_TTT = pickle.load(fn)
        else:
            self.loaded_pred_motion_params = None

        ###############
        # Correct motion corrupted undersampled k-space with all zero motion parameters which gives corrupted input to network
        with torch.no_grad():
            

            ###############
            # Log images and metrics before TTT
            if evaluate_before_TTT:
                self.evaluate_before_TTT(binary_background_mask, smaps3D=smaps3D, masked_corrupted_kspace3D=masked_corrupted_kspace3D, smaps3D_conj=smaps3D_conj,
                                        save_slices=True, save_3D = False, list_of_slices = None, traj=traj, pred_motion_params=self.loaded_pred_motion_params)
                
            # # Save one slice for each coil for the sensitivity maps and motion corrupted volume
            # and volume from which we computed sensitivity maps
            # list_of_slices = [(2, masked_motion_free_kspace3D.shape[3]//2)]
            # masked_motion_free_img3D_coil = motion_correction_NUFFT(masked_motion_free_kspace3D, -1* pred_motion_params, traj, 
            #                                                         weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
            #                                                         max_coil_size=self.args.TTT_nufft_max_coil_size)
            # for c in range(masked_motion_free_img3D_coil.shape[0]):
            #     save_slice_images_from_volume(masked_motion_free_img3D_coil[c].cpu(), list_of_slices, self.args.TTT_results_path, f"masked_motion_free_img3D_coil{c}", axis_names = ["coronal","saggital","axial"], dir_name="coil_images")
            #     save_slice_images_from_volume(masked_corrupted_img3D_coil[c].cpu(), list_of_slices, self.args.TTT_results_path, f"masked_corrupted_img3D_coil{c}", axis_names = ["coronal","saggital","axial"], dir_name="coil_images")
            #     save_slice_images_from_volume(smaps3D[c].cpu(), list_of_slices, self.args.TTT_results_path, f"smap_coil{c}", axis_names = ["coronal","saggital","axial"], dir_name="coil_images")

        return traj, smaps3D, smaps3D_conj, binary_background_mask, masked_corrupted_kspace3D, mask3D

    def evaluate_after_TTT_step(self,TTT_loss, pred_motion_params, iteration, recon_img3D, binary_background_mask, optimizer, ax_ind, rec_ind, new_phase=False, traj=None):
        # !!!
        
        self.TTT_meters_per_example["TTT_loss"].update(TTT_loss, iteration)
        self.TTT_meters_per_example["ax_inds"].update(ax_ind, iteration)

        self.pred_motion_params_over_epochs = pred_motion_params.cpu().unsqueeze(-1) if iteration==0 or new_phase else torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1)

        ### Save reconstruction at minimum dc loss
        if TTT_loss==self.TTT_meters_per_example["TTT_loss"].best_val:
            recon_img3D = complex_abs(recon_img3D * binary_background_mask)
            self.recon_img3D = recon_img3D.cpu()
            list_of_slices = None
            save_slice_images_from_volume(self.recon_img3D[0], list_of_slices, self.args.TTT_results_path, "current_best_recon_TTT", axis_names = ["coronal","saggital","axial"], dir_name=f"slice_images_phase{self.phase}")

        text_to_log = f"Epoch {iteration} | "
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            if name == "ax_inds":
                continue
            text_to_log += f"{name}: {meter.val[-1]:.5f} | "

        text_to_log += f"lr: {optimizer.param_groups[0]['lr']:.5e} | AxInd {ax_ind} | SliceInd {rec_ind}"
        logging.info(text_to_log)

        N_s = self.pred_motion_params_over_epochs.shape[0]
        pred_motion_params = pred_motion_params.cpu().numpy()

        if ax_ind == 2:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params, N_s, dc_loss_ind = -1, fig_name=None)
        if iteration % 5 == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params, N_s, dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_axind_{ax_ind}.png")
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params, N_s, dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_axind_{ax_ind}_motParams.png", plot_dc_losses=False)
        #if iteration == 5:
        #    print("Stop here")
        

    def evaluate_after_TTT(self, masked_corrupted_kspace3D=None, traj=None, smaps3D_conj=None, binary_background_mask=None, optimizer=None):

        save_path_num = os.path.join(self.args.TTT_results_path_numerical, f"phase{self.phase}")
        if not os.path.exists(save_path_num):
            os.makedirs(save_path_num)
        
        torch.save(self.recon_img3D, os.path.join(save_path_num, "reconstruction_min_reconDC_loss.pt"))
        pickle.dump( self.TTT_meters_per_example, open( os.path.join(save_path_num, 'TTT_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        np.save(os.path.join(save_path_num, 'pred_motion_params_over_epochs.npy'), self.pred_motion_params_over_epochs.cpu().numpy())
        torch.save(optimizer.state_dict(), os.path.join(save_path_num, 'optimizer.pth'))


        N_s = self.pred_motion_params_over_epochs.shape[0]
        best_step = self.TTT_meters_per_example["TTT_loss"].best_count

        # The DC loss in the 0th iter corresponds to the all zeto motion state
        # Hence, no best_step+1 is needed
        pred_motion_params_final = self.pred_motion_params_over_epochs[:,:,best_step].numpy() 

        self.final_result_dict["TTT_loss"] = self.TTT_meters_per_example["TTT_loss"].best_val
        self.final_result_dict["TTT_best_step"] = best_step 
        self.final_result_dict["pred_motion_params_final"] = pred_motion_params_final
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,best_step].cpu().numpy()
        self.final_result_dict["traj"] = traj
        self.final_result_dict["gt_motion_params"] = self.gt_motion_params


        pickle.dump( self.final_result_dict, open( os.path.join(save_path_num, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(save_path_num, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(save_path_num, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_grad_per_state.numpy(), open( os.path.join(save_path_num, 'track_grad_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        if masked_corrupted_kspace3D is not None:
            with torch.no_grad():
                list_of_slices = None
                input = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* torch.from_numpy(pred_motion_params_final), traj, weight_rot=True, args=self.args, max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                masked_corrected_img3D = complex_mul(input, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D
                recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_TTT", axis_names = ["coronal","saggital","axial"], dir_name=f"slice_images_phase{self.phase}")
                save_slice_images_from_volume(masked_corrected_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected_TTT", axis_names = ["coronal","saggital","axial"], dir_name=f"slice_images_phase{self.phase}")

        self.save_figure_motion_parameters_dc_loss(best_step, pred_motion_params_final, N_s, dc_loss_ind = best_step, fig_name=f"motion_pred_params_best_step_{best_step}_phase{self.phase}.png")
        self.save_figure_motion_parameters_dc_loss(best_step, pred_motion_params_final, N_s, dc_loss_ind = best_step, fig_name=f"motion_pred_params_best_step_{best_step}_phase{self.phase}_motParams.png", plot_dc_losses=False)

        logging.info(f"Best step (min consistency loss) in phase {self.phase}: {best_step}")
        #logging.info(f"Motion prediction at best step: {pred_motion_params_final}")
        
        # # Implement plots of dc losses per state over epochs but only where ax_ind==2
        DC_losses = np.array(self.TTT_meters_per_example["TTT_loss"].val)
        ax_inds = self.TTT_meters_per_example["ax_inds"].val
        ax_inds_axind2 = np.where(np.array(ax_inds)==2)[0]
        DC_losses_axind2 = DC_losses[ax_inds_axind2]
        steps_axind2 = np.array(self.TTT_meters_per_example["TTT_loss"].epochs)[ax_inds_axind2]
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(steps_axind2,DC_losses_axind2)
        plt.xlabel('Steps')
        plt.ylabel('DC loss')
        plt.title('DC loss over steps')
        plt.subplot(1,2,2)
        for i in range(N_s):
            plt.plot(steps_axind2,self.track_dc_losses_per_state_norm_per_state[i,np.where(np.array(ax_inds)==2)[0]].cpu().numpy(), alpha=0.1, color='blue')
        plt.xlabel('Steps')
        plt.ylabel('DC loss per state')
        plt.title('DC loss per state over steps')
        plt.savefig(self.args.TTT_results_path+f"/dc_loss_convergence_phase{self.phase}.png")

        # # Plot motion parameters over steps
        titles = ["x axis translation", "y axis translation", "z axis translation", "x-y plane rotation", "y-z plane rotation", "x-z plane rotation"]

        num_steps = self.pred_motion_params_over_epochs.shape[2]
        plt.figure(figsize=(20,10))
        for i,title in enumerate(titles):
            plt.subplot(2,3,i+1)
            for j in range(N_s):
                plt.plot(range(0,num_steps,1),self.pred_motion_params_over_epochs[j,i,:].numpy(),alpha=0.1, color='blue')
            plt.ylabel('mot params in deg or mm')
            plt.xlabel('Steps')
            plt.title(title)
        plt.savefig(self.args.TTT_results_path+f"/mot_params_convergence_phase{self.phase}.png")

    def evaluate_before_TTT(self, binary_background_mask, traj=None, masked_corrected_img3D=None, smaps3D_conj=None, masked_corrupted_kspace3D=None, smaps3D=None, pred_motion_params=None, save_slices=False, save_3D = False, list_of_slices = None):
        '''
        Run this function to evaluate the model before TTT. The function computes:
            - PSNR between reference and 
                undersampled volume
                undersampled corrupted volume
                undersampled corrected volume
            - along different axes with and without masking PSNR between reference and 
                reconstructed undersampled volume
                reconstructed undersampled corrupted volume
                reconstructed undersampled corrected volume
        '''
        with torch.no_grad():
            recon_axial = True
            recon_saggital = False
            recon_coronal = False

            if pred_motion_params is None:
                pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)

            masked_corrupted_img3D_coil = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj, 
                                                                weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                max_coil_size=self.args.TTT_nufft_max_coil_size)
            masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

            ######
            # Inspect fully sampled reference volume vs. undersampled volume

            if save_slices:
                save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrupted", axis_names = ["coronal","saggital","axial"])
                if masked_corrected_img3D is not None:
                    save_slice_images_from_volume(masked_corrected_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected", axis_names = ["coronal","saggital","axial"])
                save_slice_images_from_volume(smaps3D[0,:,:,:,:].cpu(), list_of_slices, self.args.TTT_results_path, "smap0", axis_names = ["coronal","saggital","axial"])

            if save_3D:
                volume_to_k3d_html(complex_abs(masked_corrupted_img3D).cpu().numpy(), volume_name = "masked_corrupted", save_path = self.args.TTT_results_path)
                if masked_corrected_img3D is not None:
                    volume_to_k3d_html(complex_abs(masked_corrected_img3D).cpu().numpy(), volume_name = "masked_corrected", save_path = self.args.TTT_results_path)
                volume_to_k3d_html(complex_abs(smaps3D[0]).cpu().numpy(), volume_name = "smap0", save_path = self.args.TTT_results_path)

            ########
            # Reconstruct undersampled corrupted volume with 2D network along different axes

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = unet_forward_all_axes(self.model, masked_corrupted_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                # inspect 
                # - energy per state of given corrupted kspace
                # - energy per state of reconstructed k-space
                # - difference per state of given corrupted kspace and reconstructed k-space normalized
                #       - with energy per state of given corrupted kspace
                #       - with energy per state of reconstructed k-space
                #       - not at all

                mask2D = np.zeros((recon_img3D_axial_fg.shape[1], recon_img3D_axial_fg.shape[2]))
                mask2D[np.where(masked_corrupted_kspace3D[0,:,:,0,0].cpu().numpy() != 0)] = 1
                masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D , save_path=self.args.TTT_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu)

                recon_coil = complex_mul(recon_img3D_axial.unsqueeze(0), smaps3D) # recon_img3D_coil
                recon_ksp = fft2c_ndim(recon_coil, 3) # recon_kspace3D

               
                recon_ksp = motion_corruption_NUFFT(recon_ksp, recon_coil, pred_motion_params, traj, weight_rot=True, args=self.args,
                                                grad_translate=False, grad_rotate=False, 
                                                states_with_grad=None, max_coil_size=self.args.TTT_nufft_max_coil_size) 

                energy_per_state_corrupted = torch.zeros((self.args.Ns,1))
                energy_per_state_recon = torch.zeros((self.args.Ns,1))
                diff_norm_corrupted_ksp = torch.zeros((self.args.Ns,1))
                diff_norm_recon_ksp = torch.zeros((self.args.Ns,1))
                diff_no_norm = torch.zeros((self.args.Ns,1))

                for i in range(self.args.Ns):
                    state_ksp_corrupted = masked_corrupted_kspace3D * masks2D_all_states[i]
                    state_ksp_recon = recon_ksp * masks2D_all_states[i]
                    energy_per_state_corrupted[i] = torch.sum(torch.abs(state_ksp_corrupted))
                    energy_per_state_recon[i] = torch.sum(torch.abs(state_ksp_recon))
                    diff_norm_corrupted_ksp[i] = torch.sum(torch.abs(state_ksp_corrupted - state_ksp_recon)).cpu() / energy_per_state_corrupted[i]
                    diff_norm_recon_ksp[i] = torch.sum(torch.abs(state_ksp_corrupted - state_ksp_recon)).cpu() / energy_per_state_recon[i]
                    diff_no_norm[i] = torch.sum(torch.abs(state_ksp_corrupted - state_ksp_recon))

                save_path = os.path.join(self.args.TTT_results_path, f"stats_per_state_phase{self.phase}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.figure(figsize=(25,5))
                plt.subplot(1,5,1)
                plt.plot(energy_per_state_corrupted.cpu().numpy())
                plt.title('Energy per state of corrupted ksp')
                plt.subplot(1,5,2)
                plt.plot(energy_per_state_recon.cpu().numpy())
                plt.title('Energy per state of recon ksp')
                plt.subplot(1,5,3)
                plt.plot(diff_norm_corrupted_ksp.cpu().numpy())
                plt.title('Loss per state corrupted ksp normalization')
                plt.subplot(1,5,4)
                plt.plot(diff_norm_recon_ksp.cpu().numpy())
                plt.title('Loss per state recon ksp normalization')
                plt.subplot(1,5,5)
                plt.plot(diff_no_norm.cpu().numpy())
                plt.title('Loss per state no normalization')
                plt.savefig(os.path.join(save_path, "energy_per_state_corrupted_recon_ksp.png"))
                plt.close()

                # required if we want to split motion states already in the 0-th iter
                self.track_dc_losses_per_state_norm_per_state_init = diff_norm_corrupted_ksp
                    


            # saggital reconstruction
            if recon_saggital:
                recon_img3D_saggital = unet_forward_all_axes(self.model, masked_corrupted_img3D, rec_id=None,ax_id=1)
                recon_img3D_saggital_fg = recon_img3D_saggital * binary_background_mask

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = unet_forward_all_axes(self.model, masked_corrupted_img3D, rec_id=None,ax_id=0)
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

            if save_slices:
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_axial else None
                save_slice_images_from_volume(recon_img3D_saggital_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_saggital_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_saggital else None
                save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_coronal else None

            if save_3D:
                volume_to_k3d_html(complex_abs(recon_img3D_axial_fg[0]).cpu().numpy(), volume_name = "recon_axial_corrupted_masked", save_path = self.args.TTT_results_path) if recon_axial else None
                volume_to_k3d_html(complex_abs(recon_img3D_saggital_fg[0]).cpu().numpy(), volume_name = "recon_saggital_corrupted_masked", save_path = self.args.TTT_results_path) if recon_saggital else None
                volume_to_k3d_html(complex_abs(recon_img3D_coronal_fg[0]).cpu().numpy(), volume_name = "recon_coronal_corrupted_masked", save_path = self.args.TTT_results_path) if recon_coronal else None

            ########
            # Reconstruct undersampled corrected volume with 2D network along different axes
            if masked_corrected_img3D is not None:
                if recon_axial:
                    # axial reconstruction
                    recon_img3D_axial = unet_forward_all_axes(self.model, masked_corrected_img3D, rec_id=None,ax_id=2)
                    recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                # saggital reconstruction
                if recon_saggital:
                    recon_img3D_saggital = unet_forward_all_axes(self.model, masked_corrected_img3D, rec_id=None,ax_id=1)
                    recon_img3D_saggital_fg = recon_img3D_saggital * binary_background_mask

                if recon_coronal:  
                    # coronal reconstruction
                    recon_img3D_coronal = unet_forward_all_axes(self.model, masked_corrected_img3D, rec_id=None,ax_id=0)
                    recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                if save_slices:
                    save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_axial else None
                    save_slice_images_from_volume(recon_img3D_saggital_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_saggital_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_saggital else None
                    save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_corrupted_masked", axis_names = ["coronal","saggital","axial"]) if recon_coronal else None

                if save_3D:
                    volume_to_k3d_html(complex_abs(recon_img3D_axial_fg[0]).cpu().numpy(), volume_name = "recon_axial_corrupted_masked", save_path = self.args.TTT_results_path) if recon_axial else None
                    volume_to_k3d_html(complex_abs(recon_img3D_saggital_fg[0]).cpu().numpy(), volume_name = "recon_saggital_corrupted_masked", save_path = self.args.TTT_results_path) if recon_saggital else None
                    volume_to_k3d_html(complex_abs(recon_img3D_coronal_fg[0]).cpu().numpy(), volume_name = "recon_coronal_corrupted_masked", save_path = self.args.TTT_results_path) if recon_coronal else None



    def save_figure_motion_parameters_dc_loss(self, iteration, pred_motion_params, N_s, dc_loss_ind=-1, fig_name=None, plot_dc_losses=True):

        save_dir = os.path.join(self.args.TTT_results_path, f"motion_param_figures_phase{self.phase}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        titles = ["x axis translation", "y axis translation", "z axis translation", "x-y plane rotation", "y-z plane rotation", "x-z plane rotation"]

        plt.figure(figsize=(25,10))
        for i,title in enumerate(titles):
            plt.subplot(2,3,i+1)
            plt.plot(range(0,N_s,1),pred_motion_params[:,i], alpha=0.8)
            plt.ylabel('mot params in deg or mm')
            plt.xlabel('Motion states over time')
            if i==0:
                plt.legend(['Prediction'], loc='lower left')
            if plot_dc_losses:
                plt.twinx()
                plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r', alpha=0.8)
                plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_per_state[:,0].cpu().numpy(), 'g', alpha=0.6)
                if i==0:
                    plt.legend([f"DC iter{dc_loss_ind}", 'DC iter0'], loc='lower right')
            plt.title(title+" iter "+str(iteration))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()
    
            
