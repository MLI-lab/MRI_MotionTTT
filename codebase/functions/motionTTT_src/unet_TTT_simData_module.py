import torch
import logging
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt 

from functions.utils.helpers.meters import TrackMeter
from functions.utils.helpers.helpers_math import complex_abs, complex_mul, chunks, ifft2c_ndim, fft2c_ndim

from functions.utils.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories, save_slice_images_from_volume

from functions.utils.models.helpers_model import unet_forward_all_axes
from functions.utils.motion_simulation.motion_helpers import motion_alignment, expand_mps_to_kspline_resolution
from functions.utils.motion_simulation.motion_forward_backward_models import motion_correction_NUFFT, motion_corruption_NUFFT
from functions.utils.motion_simulation.motion_trajectories import sim_motion_get_gt_motion_traj
from functions.utils.motion_simulation.sampling_trajectories import sim_motion_get_traj


from functions.utils.helpers.helpers_img_metrics import PSNR_torch

from functions.motionTTT_src.unet_TTT_base_module import UnetTTTModuleBase

from functions.utils.data.data_loaders import cc359_loader



class UnetTTTModuleSim(UnetTTTModuleBase):

    def __init__(
            self,
            args,
            model,
            ) -> None:
        super().__init__(args, model)

        self.TTT_meters_per_example = self.init_TTT_meters()

    
    def init_TTT_meters(self):
        #  Decide what quantities to track during TTT
        TTT_meters_per_example = {
            "TTT_loss" : TrackMeter('decaying'),    
            "L1_motion_parameters" : TrackMeter('decaying'),
            "PSNR_recon_ref" : TrackMeter('increasing'),
            "ax_inds" : TrackMeter('decaying'),
        } 
                
        return TTT_meters_per_example

    def load_data_init_motion(self, evaluate_before_TTT):

        ref_img3D, mask3D, masked_kspace3D, smaps3D_conj, ref_kspace3D, ref_img3D_coil, binary_background_mask, smaps3D = cc359_loader(self.args) 
        self.ref_img3D = ref_img3D

        self.phase = 0

        ###############
        # Generate sampling trajectory
        traj = sim_motion_get_traj(self.args, mask3D)

        if self.args.Ns == self.args.num_shots:
            self.ksp_lines_per_shot = [len(traj[0][i]) for i in range(len(traj[0]))]
        else:
            raise ValueError("Currently motionTTT only supports Ns == num_shots.")
        
        ###############
        # Generate motion free undersampled k-space
        if self.args.TTT_use_nufft_with_dcomp:
            masked_img3D_coil = motion_correction_NUFFT(masked_kspace3D, torch.zeros(self.args.Ns, 6).cuda(self.args.gpu), traj, weight_rot=True, args=self.args,
                                                        do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3)
        else:
            masked_img3D_coil = ifft2c_ndim(masked_kspace3D, 3)
        masked_img3D = complex_mul(masked_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        
        ###############
        # Generate ground truth Motion trajectory. Both gt_motion_params and gt_traj have per k-space line resolution
        self.gt_motion_params, self.gt_traj, _ = sim_motion_get_gt_motion_traj(self.args, traj)

        ###############
        # Motion artifact simulation:
        # Reduce the number of motion states by combining motion states with the same motion parameters to save some time here
        gt_motion_params_combined = self.gt_motion_params[0:1,:]
        gt_traj_combined = ([self.gt_traj[0][0]], [self.gt_traj[1][0]])
        for i in range(1, self.gt_motion_params.shape[0]):
            if torch.sum(torch.abs(self.gt_motion_params[i]-self.gt_motion_params[i-1])) > 0:
                gt_motion_params_combined = torch.cat((gt_motion_params_combined, self.gt_motion_params[i:i+1,:]), dim=0)
                gt_traj_combined[0].append(self.gt_traj[0][i]) 
                gt_traj_combined[1].append(self.gt_traj[1][i])
            else:
                gt_traj_combined[0][-1] = np.concatenate((gt_traj_combined[0][-1], self.gt_traj[0][i]), axis=0)
                gt_traj_combined[1][-1] = np.concatenate((gt_traj_combined[1][-1], self.gt_traj[1][i]), axis=0)

        masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, ref_img3D_coil, gt_motion_params_combined, gt_traj_combined, weight_rot=True, args=self.args,
                                                            max_coil_size=self.args.TTT_nufft_max_coil_size)
        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        

        ###############
        # Correct motion corrected undersampled k-space with gt motion parameters
        masked_corrected_img3D_coil = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* gt_motion_params_combined, gt_traj_combined, 
                                                              weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                              max_coil_size=self.args.TTT_nufft_max_coil_size)
        masked_corrected_img3D = complex_mul(masked_corrected_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        ###############
        # Log images and metrics before TTT
        if evaluate_before_TTT:
            self.evaluate_before_TTT(masked_corrupted_img3D, binary_background_mask, 
                                    masked_corrected_img3D, masked_img3D, smaps3D=smaps3D, 
                                    smaps3D_conj=smaps3D_conj, masked_corrupted_kspace3D=masked_corrupted_kspace3D, traj=traj,
                                    save_slices=True, save_3D = False, list_of_slices = None, pred_motion_params=None,
                                    masked_corrected_gt_img3D_coil=masked_corrected_img3D_coil)

        return traj, smaps3D, smaps3D_conj, binary_background_mask, masked_corrupted_kspace3D, mask3D




    def evaluate_after_TTT_step(self,TTT_loss, pred_motion_params, iteration, recon_img3D, binary_background_mask, optimizer, ax_ind, rec_ind, new_phase=False, traj=None):
        
        self.TTT_meters_per_example["TTT_loss"].update(TTT_loss, iteration)
        self.TTT_meters_per_example["ax_inds"].update(ax_ind, iteration)

        # gt_motion_params are per k-space line resolution -> expand pred_motion_params to k-space line resolution
        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,0], self.track_dc_losses_per_state_norm_per_state[:,-1]]
        pred_mp_streched, list_of_track_dc_losses_aligned, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=list_of_track_dc_losses)

        L1_motion_parameters = torch.sum(torch.abs(pred_mp_streched-self.gt_motion_params))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        self.TTT_meters_per_example["L1_motion_parameters"].update(L1_motion_parameters, iteration)
     
        self.pred_motion_params_over_epochs = pred_motion_params.cpu().unsqueeze(-1) if iteration==0 or new_phase else torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1)

        if recon_img3D is not None:
            recon_img3D = complex_abs(recon_img3D * binary_background_mask)

            self.TTT_meters_per_example["PSNR_recon_ref"].update(PSNR_torch(recon_img3D, self.ref_img3D), iteration)

            ### Save reconstruction at minimum dc loss
            if TTT_loss==self.TTT_meters_per_example["TTT_loss"].best_val:
                self.recon_img3D = recon_img3D.cpu()
                list_of_slices = None
                save_slice_images_from_volume(self.recon_img3D[0], list_of_slices, self.args.TTT_results_path, "current_best_recon_TTT", axis_names = ["coronal","sagittal","axial"], dir_name=f"slice_images_phase{self.phase}")

        else:
            self.TTT_meters_per_example["PSNR_recon_ref"].update(0, iteration)

        if iteration % 10 == 0:
            # Compute motion error after alignment
            pred_mp_streched_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
            L1_motion_parameters_aligned = torch.sum(torch.abs(pred_mp_streched_aligned.cpu()-self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        else:
            L1_motion_parameters_aligned = 0

        
        text_to_log = f"Step {iteration} | "
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            if name == "ax_inds":
                continue
            text_to_log += f"{name}: {meter.val[-1]:.5f} | "

        text_to_log += f"L1_motion_parameters_aligned: {L1_motion_parameters_aligned:.5f} | lr: {optimizer.param_groups[0]['lr']:.5e} | AxInd {ax_ind} | SliceInd {rec_ind}"
        logging.info(text_to_log)

        #if ax_ind == 2:
        #    self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], list_of_track_dc_losses_aligned[1], dc_loss_ind = -1, fig_name=None)
        if iteration % 20 == 0 and ax_ind == 2:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], list_of_track_dc_losses_aligned[1], dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_axind_{ax_ind}.png")
            self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], list_of_track_dc_losses_aligned[1], dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_axind_{ax_ind}_motParams.png", plot_dc_losses=False)

        

    def evaluate_after_TTT(self, masked_corrupted_kspace3D=None, traj=None, smaps3D_conj=None, binary_background_mask=None, optimizer=None):

        save_path_num = os.path.join(self.args.TTT_results_path_numerical, f"phase{self.phase}")
        if not os.path.exists(save_path_num):
            os.makedirs(save_path_num)
        
        pickle.dump( self.TTT_meters_per_example, open( os.path.join(save_path_num, 'TTT_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        np.save(os.path.join(save_path_num, 'pred_motion_params_over_epochs.npy'), self.pred_motion_params_over_epochs.cpu().numpy())
        torch.save(self.recon_img3D, os.path.join(save_path_num,"reconstruction_min_reconDC_loss.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_path_num, 'optimizer.pth'))
        
        N_s = self.pred_motion_params_over_epochs.shape[0]
        best_step = self.TTT_meters_per_example["TTT_loss"].best_count

        # The DC loss in the 0th iter corresponds to the all zeto motion state
        pred_motion_params_final = self.pred_motion_params_over_epochs[:,:,best_step].numpy() 

        self.final_result_dict["TTT_loss"] = self.TTT_meters_per_example["TTT_loss"].best_val
        self.final_result_dict["TTT_best_step"] = best_step 
        self.final_result_dict["L1_motion_parameters"] = self.TTT_meters_per_example["L1_motion_parameters"].val[best_step]
        self.final_result_dict["PSNR_ref_vs_recon_TTT"] = self.TTT_meters_per_example["PSNR_recon_ref"].val[best_step]
        self.final_result_dict["pred_motion_params_final"] = pred_motion_params_final
        self.final_result_dict["gt_motion_params"] = self.gt_motion_params.cpu().numpy()
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,best_step].cpu().numpy()
        self.final_result_dict["traj"] = traj
        self.final_result_dict["gt_traj"] = self.gt_traj
        
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(save_path_num, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(save_path_num, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,0], self.track_dc_losses_per_state_norm_per_state[:,best_step]]
        pred_mp_streched, list_of_track_dc_losses_aligned, reduce_indicator = expand_mps_to_kspline_resolution(torch.from_numpy(pred_motion_params_final).cuda(self.args.gpu), traj, list_of_track_dc_losses=list_of_track_dc_losses)


        if masked_corrupted_kspace3D is not None:
            with torch.no_grad():
                list_of_slices = None
                input = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* torch.from_numpy(pred_motion_params_final).cuda(self.args.gpu), traj, weight_rot=True, args=self.args,
                                                max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                masked_corrected_img3D = complex_mul(input, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D
                recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_TTT", axis_names = ["coronal","sagittal","axial"], dir_name=f"slice_images_phase{self.phase}")
                save_slice_images_from_volume(complex_abs(masked_corrected_img3D).cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected_TTT", axis_names = ["coronal","sagittal","axial"], dir_name=f"slice_images_phase{self.phase}")

                # # Results after alignment
                pred_motion_params_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
                
                ## !!!
                L1_motion_parameters_aligned = torch.sum(torch.abs(pred_motion_params_aligned.cpu()-self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
                self.final_result_dict["L1_motion_parameters_aligned"] = L1_motion_parameters_aligned

                # Reduce aligned pred_motion_params to original resolution
                reduce_indicator_shifted = torch.zeros_like(reduce_indicator)
                reduce_indicator_shifted[0] = reduce_indicator[0]-1
                reduce_indicator_shifted[1:] = reduce_indicator[:-1]
                difference = reduce_indicator - reduce_indicator_shifted
                reduce_indices = torch.where(difference != 0)[0]
                pred_motion_params_aligned = pred_motion_params_aligned[reduce_indices]


                input = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params_aligned.cuda(self.args.gpu), traj, weight_rot=True, args=self.args,
                                max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                masked_corrected_img3D = complex_mul(input, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D
                recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_TTT_aligned", axis_names = ["coronal","sagittal","axial"], dir_name=f"slice_images_phase{self.phase}")
                save_slice_images_from_volume(complex_abs(masked_corrected_img3D).cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected_TTT_aligned", axis_names = ["coronal","sagittal","axial"], dir_name=f"slice_images_phase{self.phase}")
                PSNR_recon_ref_aligned = PSNR_torch(complex_abs(recon_img3D_axial_fg), self.ref_img3D)
                self.final_result_dict["PSNR_ref_vs_recon_TTT_aligned"] = PSNR_recon_ref_aligned

        self.save_figure_motion_parameters_dc_loss(best_step, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], list_of_track_dc_losses_aligned[1], dc_loss_ind = best_step, fig_name=f"motion_pred_params_best_step_{best_step}_phase{self.phase}.png")
        self.save_figure_motion_parameters_dc_loss(best_step, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], list_of_track_dc_losses_aligned[1], dc_loss_ind = best_step, fig_name=f"motion_pred_params_best_step_{best_step}_phase{self.phase}_motParams.png", plot_dc_losses=False)

        logging.info(f"Best step (min consistency loss) in phase {self.phase}: {best_step}")
        logging.info(f"L1 motion parameters at best step: {self.TTT_meters_per_example['L1_motion_parameters'].val[best_step]}")
        logging.info(f"L1 motion parameters aligned at best step: {L1_motion_parameters_aligned}")
        logging.info(f"PSNR reference vs. recon at best step: {self.TTT_meters_per_example['PSNR_recon_ref'].val[best_step]}")


        logging.info(f"Best L1 motion parameters: {self.TTT_meters_per_example['L1_motion_parameters'].best_val} at step {self.TTT_meters_per_example['L1_motion_parameters'].best_epoch}")
        logging.info(f"Best PSNR reference vs. recon: {self.TTT_meters_per_example['PSNR_recon_ref'].best_val} at step {self.TTT_meters_per_example['PSNR_recon_ref'].best_epoch}")

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
        plt.close()

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
        plt.close()

        pickle.dump(self.final_result_dict, open( os.path.join(save_path_num, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

    def evaluate_before_TTT(self, masked_corrupted_img3D, binary_background_mask, masked_corrected_img3D=None, masked_img3D=None, 
                            smaps3D=None, smaps3D_conj=None, masked_corrupted_kspace3D=None, traj=None, save_slices=False, 
                            save_3D = False, list_of_slices = None, pred_motion_params=None, masked_corrected_gt_img3D_coil=None):
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
            recon_sagittal = False
            recon_coronal = False

            if pred_motion_params is None:
                pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)

            ######
            # Compute DC loss of unet recon under ground truth motion
            # to get a feeling for how far off the initial DC loss of the corrupted k-space is

            masked_corrected_gt_img3D = complex_mul(masked_corrected_gt_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

            recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrected_gt_img3D, rec_id=None,ax_id=2)
            recon_coil = complex_mul(recon_img3D_axial.unsqueeze(0), smaps3D)
            recon_ksp = fft2c_ndim(recon_coil, 3)
            recon = motion_corruption_NUFFT(recon_ksp, recon_coil, self.gt_motion_params, self.gt_traj, weight_rot=True, args=self.args,
                                                grad_translate=False, grad_rotate=False, 
                                                max_coil_size=self.args.TTT_nufft_max_coil_size) 
            loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            logging.info("\nEvaluate before MotionTTT")
            logging.info(f"DC loss of unet recon under gt motion correction and corruption: {loss:.5f}")


            ######
            # Inspect fully sampled reference volume vs. undersampled volume

            if save_slices:
                save_slice_images_from_volume(self.ref_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "ref", axis_names = ["coronal","sagittal","axial"])
                if masked_img3D is not None:
                    save_slice_images_from_volume(masked_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked", axis_names = ["coronal","sagittal","axial"])
                save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrupted", axis_names = ["coronal","sagittal","axial"])
                save_slice_images_from_volume(masked_corrected_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected", axis_names = ["coronal","sagittal","axial"])

            if masked_img3D is not None:
                psnr = PSNR_torch(complex_abs(masked_img3D), complex_abs(masked_corrected_img3D))
                logging.info(f"PSNR undersampled vs. undersampled corrected: {psnr}")
                self.final_result_dict["PSNR_zf_motfree_vs_zf_corrected_gtmot"] = psnr

                psnr = PSNR_torch(complex_abs(masked_img3D), self.ref_img3D)
                logging.info(f"PSNR reference vs. undersampled: {psnr}")
                self.final_result_dict["PSNR_reference_vs_zf_motfree"] = psnr

            psnr = PSNR_torch(complex_abs(masked_corrupted_img3D), self.ref_img3D)
            logging.info(f"PSNR reference vs. undersampled corrupted: {psnr}")
            self.final_result_dict["PSNR_reference_vs_zf_corrupted"] = psnr

            psnr = PSNR_torch(complex_abs(masked_corrected_img3D), self.ref_img3D)
            logging.info(f"PSNR reference vs. undersampled corrected: {psnr}")
            self.final_result_dict["PSNR_reference_vs_zf_corrected"] = psnr


            ######
            # Reconstruct undersampled volume with 2D network along different axes

            if masked_img3D is not None:
                if recon_axial:
                    # axial reconstruction
                    recon_img3D_axial = unet_forward_all_axes(self.model,masked_img3D, rec_id=None,ax_id=2)
                    recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                    psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), self.ref_img3D)
                    logging.info(f"PSNR reference vs. U-net recon axial undersampled: {psnr}")
                    self.final_result_dict["PSNR_reference_vs_recon_motfree_axial_binmasked"] = psnr

                if recon_sagittal:
                    # sagittal reconstruction
                    recon_img3D_sagittal = unet_forward_all_axes(self.model,masked_img3D, rec_id=None,ax_id=1)
                    recon_img3D_sagittal_fg = recon_img3D_sagittal * binary_background_mask

                    psnr = PSNR_torch(complex_abs(recon_img3D_sagittal_fg), self.ref_img3D)
                    logging.info(f"PSNR reference vs. U-net recon sagittal undersampled: {psnr}")
                    self.final_result_dict["PSNR_reference_vs_recon_motfree_sagittal_binmasked"] = psnr

                if recon_coronal:  
                    # coronal reconstruction
                    recon_img3D_coronal = unet_forward_all_axes(self.model,masked_img3D, rec_id=None,ax_id=0)
                    recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                    psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), self.ref_img3D)
                    logging.info(f"PSNR reference vs. U-net recon coronal undersampled: {psnr}")
                    self.final_result_dict["PSNR_reference_vs_recon_motfree_coronal_binmasked"] = psnr

                if save_slices:
                    save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_masked", axis_names = ["coronal","sagittal","axial"]) if recon_axial else None
                    save_slice_images_from_volume(recon_img3D_sagittal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_sagittal_masked", axis_names = ["coronal","sagittal","axial"]) if recon_sagittal else None
                    save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_masked", axis_names = ["coronal","sagittal","axial"]) if recon_coronal else None

            ########
            # Reconstruct undersampled corrupted volume with 2D network along different axes

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrupted_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon axial undersampled corrupted: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_axial_binmasked"] = psnr

                # inspect 
                # - energy per state of given corrupted kspace
                # - energy per state of reconstructed k-space
                # - difference per state of given corrupted kspace and reconstructed k-space normalized
                #       - with energy per state of given corrupted kspace
                #       - with energy per state of reconstructed k-space
                #       - not at all

                mask2D = np.zeros((recon_img3D_axial_fg.shape[1], recon_img3D_axial_fg.shape[2]))
                mask2D[np.where(masked_corrupted_kspace3D[0,:,:,0,0].cpu().numpy() != 0)] = 1
                masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D , save_path=self.args.TTT_results_path, dir_name="motion_sampling_traj", save_figures = False, verbose=False)).cuda(self.args.gpu)

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

            # sagittal reconstruction
            if recon_sagittal:
                recon_img3D_sagittal = unet_forward_all_axes(self.model,masked_corrupted_img3D, rec_id=None,ax_id=1)
                recon_img3D_sagittal_fg = recon_img3D_sagittal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_sagittal_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon sagittal undersampled corrupted: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_sagittal_binmasked"] = psnr

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = unet_forward_all_axes(self.model,masked_corrupted_img3D, rec_id=None,ax_id=0)
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon coronal undersampled corrupted: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_coronal_binmasked"] = psnr

            if save_slices:
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_corrupted_masked", axis_names = ["coronal","sagittal","axial"]) if recon_axial else None
                save_slice_images_from_volume(recon_img3D_sagittal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_sagittal_corrupted_masked", axis_names = ["coronal","sagittal","axial"]) if recon_sagittal else None
                save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_corrupted_masked", axis_names = ["coronal","sagittal","axial"]) if recon_coronal else None

            ########
            # Reconstruct undersampled corrected volume with 2D network along different axes

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=2)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon axial undersampled corrected: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_axial_binmasked"] = psnr

            if recon_sagittal:
                # sagittal reconstruction
                recon_img3D_sagittal = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=1)
                recon_img3D_sagittal_fg = recon_img3D_sagittal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_sagittal_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon sagittal undersampled corrected: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_sagittal_binmasked"] = psnr

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = unet_forward_all_axes(self.model,masked_corrected_img3D, rec_id=None,ax_id=0)
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), self.ref_img3D)
                logging.info(f"PSNR reference vs. U-net recon coronal undersampled corrected: {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_coronal_binmasked"] = psnr

            if save_slices:
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_corrected_masked", axis_names = ["coronal","sagittal","axial"]) if recon_axial else None
                save_slice_images_from_volume(recon_img3D_sagittal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_sagittal_corrected_masked", axis_names = ["coronal","sagittal","axial"]) if recon_sagittal else None
                save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_corrected_masked", axis_names = ["coronal","sagittal","axial"]) if recon_coronal else None


    def save_figure_motion_parameters_dc_loss(self, iteration, pred_mp, gt_mp, dc_losses_1, dc_losses_2, dc_loss_ind, fig_name=None, plot_dc_losses = True):

        save_dir = os.path.join(self.args.TTT_results_path, f"motion_param_figures_phase{self.phase}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        titles = ["x axis translation", "y axis translation", "z axis translation", "x-y plane rotation", "y-z plane rotation", "x-z plane rotation"]

        N_s = pred_mp.shape[0]
        plt.figure(figsize=(25,10))
        for i,title in enumerate(titles):
            plt.subplot(2,3,i+1)
            plt.plot(range(0,N_s,1),pred_mp[:,i], alpha=0.8)
            plt.plot(range(0,N_s,1),gt_mp[:,i], alpha=0.8)
            plt.ylabel('mot params in deg or mm')
            plt.xlabel('k-space line time index')
            if i==0:
                plt.legend(['Prediction', 'Ground truth'], loc='lower left')
            if plot_dc_losses:
                plt.twinx()
                plt.plot(range(0,N_s,1),dc_losses_2.cpu().numpy(), 'r', alpha=0.6)
                plt.plot(range(0,N_s,1),dc_losses_1.cpu().numpy(), 'g', alpha=0.6)
                if i==0:
                    plt.legend([f"DC iter{dc_loss_ind}", 'DC iter0'], loc='lower right')
            plt.title(title+" iter "+str(iteration))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()


    
            
