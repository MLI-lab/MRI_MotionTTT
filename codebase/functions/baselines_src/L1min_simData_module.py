
import torch
import numpy as np
import logging
import os
import pickle
import matplotlib.pyplot as plt

from functions.utils.helpers.meters import TrackMeter
from functions.baselines_src.L1min_base_module import L1minModuleBase
from functions.utils.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim
from functions.utils.motion_simulation.motion_helpers import expand_mps_to_kspline_resolution
from functions.utils.motion_simulation.motion_forward_backward_models import motion_corruption_NUFFT
from functions.utils.motion_simulation.motion_trajectories import sim_motion_get_gt_motion_traj
from functions.utils.motion_simulation.sampling_trajectories import sim_motion_get_traj
from functions.utils.helpers.helpers_img_metrics import PSNR_torch
from functions.utils.helpers.helpers_log_save_image_utils import save_slice_images_from_volume

from functions.utils.data.data_loaders import cc359_loader

class L1minModuleSim(L1minModuleBase):

    def __init__(
            self,
            args,
            ) -> None:
        super().__init__(args)

        self.L1min_meters = self.init_L1min_meters()

    def init_L1min_meters(self):
        #  Decide what quantities to track during L1 minimization
        L1min_meters = {
            "loss_total" : TrackMeter('decaying'),    
            "loss_dc" : TrackMeter('decaying'),
            "loss_reg" : TrackMeter('decaying'),
            "PSNR_recon_ref" : TrackMeter('increasing'),
        } 
                
        return L1min_meters      
    
    def load_data_init_motion(self):
        
        ref_img3D, mask3D, masked_kspace3D, smaps3D_conj, ref_kspace3D, ref_img3D_coil, binary_background_mask, smaps3D = cc359_loader(self.args) 
        self.ref_img3D = ref_img3D

        ###############
        # Generate sampling trajectory
        traj = sim_motion_get_traj(self.args, mask3D)

        if self.args.Ns == self.args.num_shots:
            self.ksp_lines_per_shot = [len(traj[0][i]) for i in range(len(traj[0]))]
        else:
            raise ValueError("Currently motionTTT only supports Ns == num_shots.")
        
        ###############
        # Generate motion free undersampled k-space
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
                                                            max_coil_size=self.args.L1min_nufft_max_coil_size)
        
        self.evaluate_before_L1min(masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask)

        return traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D
    
    def evaluate_before_L1min(self,masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask):

        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        list_of_slices = None        
        save_slice_images_from_volume(ref_img3D.cpu(), list_of_slices, self.args.L1min_results_path, "ref_img", axis_names = ["coronal","sagittal","axial"])
        save_slice_images_from_volume(masked_img3D.cpu(), list_of_slices, self.args.L1min_results_path, "masked_img", axis_names = ["coronal","sagittal","axial"])
        save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.L1min_results_path, "masked_corrupted_img", axis_names = ["coronal","sagittal","axial"])

        psnr = PSNR_torch(complex_abs(masked_img3D), ref_img3D)
        logging.info(f"PSNR reference vs. undersampled: {psnr}")
        self.final_result_dict["PSNR_reference_vs_zf_motfree"] = psnr

        psnr = PSNR_torch(complex_abs(masked_corrupted_img3D), ref_img3D)
        logging.info(f"PSNR reference vs. undersampled corrupted: {psnr}")
        self.final_result_dict["PSNR_reference_vs_zf_corrupted"] = psnr

    def evaluate_after_L1min_step(self, recon, pred_motion_params, traj, loss_recon, loss_reg, loss_dc, iteration):

        self.L1min_meters["loss_total"].update(loss_recon, iteration)
        self.L1min_meters["loss_dc"].update(loss_dc, iteration)
        self.L1min_meters["loss_reg"].update(loss_reg, iteration)
            
        psnr_step = PSNR_torch(complex_abs(recon), self.ref_img3D)
        self.L1min_meters["PSNR_recon_ref"].update(psnr_step, iteration)

        logging.info(f"iteration {iteration} -- recon loss: {loss_recon:.5f} | DC loss: {loss_dc:.5f} | Reg loss: {loss_reg:.5f} | PSNR: {psnr_step:.5f}")
        
        list_of_slices = None
        if iteration % 15 == 0:
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.L1min_results_path, f"recon_iter_{iteration}", axis_names = ["coronal","sagittal","axial"], dir_name="recons_per_iter")

        ### Save reconstruction at minimum dc loss
        if loss_dc==self.L1min_meters["loss_dc"].best_val:
            self.reconstruction_min_reconDC_loss = recon

        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,-1]]
        pred_mp_streched, list_of_track_dc_losses_aligned, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=list_of_track_dc_losses)
            
        if iteration % 15 == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')
            self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_motParams.png", plot_dc_losses=False)

    def evaluate_after_L1min(self, traj, pred_motion_params):

        torch.save(self.reconstruction_min_reconDC_loss.cpu(), self.args.L1min_results_path+"/reconstruction_min_reconDC_loss.pt")
        pickle.dump(self.L1min_meters, open( os.path.join(self.args.L1min_results_path, 'L1min_meters.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.L1min_results_path, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.L1min_results_path, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        # load the best reconstruction according to DC loss
        best_step = self.L1min_meters["loss_dc"].best_epoch
        best_reconDC_loss = self.L1min_meters["loss_dc"].best_val
        psnr = PSNR_torch(complex_abs(self.reconstruction_min_reconDC_loss), self.ref_img3D)
        logging.info(f"PSNR of the best reconstruction according to min recon DC loss: {psnr} at step {best_step}")

        list_of_slices = None
        save_slice_images_from_volume(self.reconstruction_min_reconDC_loss[0].cpu(), list_of_slices, self.args.L1min_results_path, "recon_min_reconDC_loss", axis_names = ["coronal","sagittal","axial"])

        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,best_step]]
        pred_mp_streched, list_of_track_dc_losses_aligned, reduce_indicator = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=list_of_track_dc_losses)

        self.final_result_dict["best_reconDC_loss"] = best_reconDC_loss
        self.final_result_dict["best_reconDC_step"] = best_step 
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,best_step].cpu().numpy()
        self.final_result_dict["PSNR_ref_vs_recon"] = psnr
        self.final_result_dict["pred_motion_params"] = pred_motion_params.cpu().numpy()
        self.final_result_dict["gt_motion_params"] = self.gt_motion_params.cpu().numpy()
        self.final_result_dict["traj"] = traj
        self.final_result_dict["gt_traj"] = self.gt_traj

        pickle.dump(self.final_result_dict, open( os.path.join(self.args.L1min_results_path, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        self.save_figure_motion_parameters_dc_loss(best_step, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind=best_step, fig_name=f'motion_pred_params_min_reconDC_loss_totstep_{best_step}.png', )

        DC_losses = self.L1min_meters["loss_dc"].val
        steps = self.L1min_meters["loss_dc"].epochs
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(steps,DC_losses)
        plt.xlabel('Steps')
        plt.ylabel('DC loss')
        plt.title('DC loss over steps')
        plt.subplot(1,2,2)
        for i in range(self.track_dc_losses_per_state_norm_per_state.shape[0]):
            plt.plot(steps,self.track_dc_losses_per_state_norm_per_state[i,:].cpu().numpy(), alpha=0.1, color='blue')
        plt.xlabel('Steps')
        plt.ylabel('DC loss per state')
        plt.title('DC loss per state over steps')
        plt.savefig(self.args.L1min_results_path+"/dc_loss_convergence.png")
        plt.close()

    def save_figure_motion_parameters_dc_loss(self, iteration, pred_mp, gt_mp, dc_losses_1, dc_loss_ind=-1, fig_name=None, plot_dc_losses = True):

        save_dir = os.path.join(self.args.L1min_results_path, "motion_param_figures")
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
            if i == 0:
                plt.legend(['Prediction', 'Ground truth'], loc='lower left')
            if plot_dc_losses:
                plt.twinx()
                plt.plot(range(0,N_s,1),dc_losses_1.cpu().numpy(), 'r', alpha=0.6)
                if i == 0:
                    plt.legend([f"DC iter{dc_loss_ind}"], loc='lower right')
            plt.title(title+" iter "+str(iteration))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()