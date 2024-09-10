import torch
import logging
#import pickle5 as pickle
import pickle
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from functions.helpers.meters import TrackMeter
from functions.helpers.helpers_math import complex_abs, complex_mul, chunks, ifft2c_ndim, fft2c_ndim, complex_conj

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import  motion_alignment, sim_motion_get_gt_motion_traj, expand_mps_to_kspline_resolution
from functions.motion_simulation.motion_functions import sim_motion_get_traj, motion_corruption_NUFFT


from functions.helpers.helpers_img_metrics import PSNR_torch
from torch.autograd import Variable
import ptwt, pywt

from functions.test_time_training.alt_opt_base_module import AltOptModuleBase

from functions.data.data_loaders import cc359_loader


class AltOptModule(AltOptModuleBase):

    def __init__(
            self,
            args,
            ) -> None:
        super().__init__(args)

        self.alt_opt_meters_per_example = self.init_altopt_meters()

    def init_altopt_meters(self):
        #  Decide what quantities to track during alt opt
        alt_opt_meters_per_example = {
            "recon_loss_total" : TrackMeter('decaying'),    
            "recon_loss_dc" : TrackMeter('decaying'),
            "recon_loss_reg" : TrackMeter('decaying'),
            "motion_loss" : TrackMeter('decaying'),
            "dc_loss_recon_and_motion" : TrackMeter('decaying'),
            "L1_gt_motion_parameters" : TrackMeter('decaying'),
            "PSNR_recon_ref" : TrackMeter('increasing'),
        } 
                
        return alt_opt_meters_per_example        

    def load_data_init_motion(self):
        
        ref_img3D, mask3D, masked_kspace3D, smaps3D_conj, ref_kspace3D, ref_img3D_coil, binary_background_mask, smaps3D = cc359_loader(self.args) 
        self.ref_img3D = ref_img3D

        self.phase = 0

        ###############
        # Generate sampling trajectory
        traj = sim_motion_get_traj(self.args, mask3D)

        if self.args.Ns == self.args.TTT_num_shots:
            self.ksp_lines_per_shot = [len(traj[0][i]) for i in range(len(traj[0]))]
        else:
            raise ValueError("Currently motionTTT only supports Ns == TTT_num_shots.")
        
        ###############
        # Generate motion free undersampled k-space
        masked_img3D_coil = ifft2c_ndim(masked_kspace3D, 3)
        masked_img3D = complex_mul(masked_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        
        ###############
        # Generate ground truth Motion trajectory. Both gt_motion_params and gt_traj have per k-space line resolution
        self.gt_motion_params, self.gt_traj, intraShot_event_inds = sim_motion_get_gt_motion_traj(self.args, traj)

        ###############
        # Initialize predicted motion parameters
        if self.args.TTT_intraShot_estimation_only:
            logging.info("Estimate motion parameters for intra-shot motion only. Initializing all other motion states with ground truth.")
            # Assign gt motion parameters to all states except where intra-shot motion occurs
            # Motion states there are split and initialized with zeros
            pred_motion_params = self.gt_motion_params.clone()

            # Get reduce_indicator that maps from k-space resolution to shot resolution
            _,_,reduce_indicator = expand_mps_to_kspline_resolution(torch.zeros((self.args.TTT_num_shots,6)), traj, list_of_track_dc_losses=None)

            # Reduce pred_motion_params
            reduce_indicator_shifted = torch.zeros_like(reduce_indicator)
            reduce_indicator_shifted[0] = reduce_indicator[0]-1
            reduce_indicator_shifted[1:] = reduce_indicator[:-1]
            difference = reduce_indicator - reduce_indicator_shifted
            reduce_indices = torch.where(difference != 0)[0]
            pred_motion_params = pred_motion_params[reduce_indices]

            # Update the number of motion states
            self.args.Ns = self.args.Ns + len(intraShot_event_inds) * (self.args.TTT_states_per_split-1)
            self.phase = 1
            logging.info(f"New number of states after splitting intra-shot affected shots into {self.args.TTT_states_per_split} states per shot: {self.args.Ns}")

            # Split motion parameters where intra-shot motion occurs and init with zeros. Update traj.
            # Set up list with indices of states that we want to compute gradients for
            # (we continue to assume that no intra-shot motion occurs in the first shot)            
            self.loaded_pred_motion_params = pred_motion_params[0:1,:]
            traj_split = (traj[0][0:1], traj[1][0:1])
            i_offset = 0
            self.Ns_list_after_split = []
            for i in range(1,self.args.TTT_num_shots):
                if i in intraShot_event_inds:
                    self.loaded_pred_motion_params = torch.cat((self.loaded_pred_motion_params, torch.zeros(self.args.TTT_states_per_split,6).cuda(self.args.gpu)), dim=0)
                    traj_split[0].extend(list(chunks(traj[0][i], self.args.TTT_states_per_split)))
                    traj_split[1].extend(list(chunks(traj[1][i], self.args.TTT_states_per_split)))
                    self.Ns_list_after_split.extend([i+i_offset+j for j in range(self.args.TTT_states_per_split)])
                    i_offset += self.args.TTT_states_per_split-1
                else:
                    self.loaded_pred_motion_params = torch.cat((self.loaded_pred_motion_params, pred_motion_params[i:i+1,:]), dim=0)
                    traj_split[0].append(traj[0][i])
                    traj_split[1].append(traj[1][i])

            assert self.loaded_pred_motion_params.shape[0] == self.args.Ns
            assert len(traj_split[0]) == self.args.Ns
            traj = traj_split

            logging.info(f"State indices for which we compute gradients after splitting: {self.Ns_list_after_split} in total {len(self.Ns_list_after_split)}")
            logging.info(f"TTT_all_states_grad_after_split is {self.args.TTT_all_states_grad_after_split}")
        else:
            self.loaded_pred_motion_params = None

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
        
        self.evaluate_before_altopt(masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask)

        return traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D
                    
    def evaluate_before_altopt(self,masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask):

        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        list_of_slices = None        
        save_slice_images_from_volume(ref_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "ref_img", axis_names = ["coronal","saggital","axial"])
        save_slice_images_from_volume(masked_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_img", axis_names = ["coronal","saggital","axial"])
        save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_corrupted_img", axis_names = ["coronal","saggital","axial"])

        psnr = PSNR_torch(complex_abs(masked_img3D), ref_img3D)
        logging.info(f"PSNR reference vs. undersampled: {psnr}")
        self.final_result_dict["PSNR_reference_vs_zf_motfree"] = psnr

        psnr = PSNR_torch(complex_abs(masked_corrupted_img3D), ref_img3D)
        logging.info(f"PSNR reference vs. undersampled corrupted: {psnr}")
        self.final_result_dict["PSNR_reference_vs_zf_corrupted"] = psnr

        # save k-space slices
        #save_slice_images_from_volume(ref_kspace3D.cpu(), list_of_slices, self.args.altopt_results_path, "ref_ksp", axis_names = ["coronal","saggital","axial"],kspace=True,coil_index=0)
        #save_slice_images_from_volume(masked_kspace3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_ksp", axis_names = ["coronal","saggital","axial"],kspace=True,coil_index=0)
        #save_slice_images_from_volume(masked_corrupted_kspace3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_corrupted_ksp", axis_names = ["coronal","saggital","axial"],kspace=True,coil_index=0)


    def evaluate_after_recon_step(self, recon, pred_motion_params, traj, loss_recon, loss_reg, loss_dc, iteration, recon_step, total_steps):

        self.alt_opt_meters_per_example["recon_loss_total"].update(loss_recon, total_steps)
        self.alt_opt_meters_per_example["recon_loss_dc"].update(loss_dc, total_steps)
        self.alt_opt_meters_per_example["recon_loss_reg"].update(loss_reg, total_steps)
        self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].update(loss_dc, total_steps)

            
        psnr_step = PSNR_torch(complex_abs(recon), self.ref_img3D)
        self.alt_opt_meters_per_example["PSNR_recon_ref"].update(psnr_step, total_steps)

        logging.info(f"total step {total_steps}, iteration {iteration}, recon step {recon_step} -- recon loss: {loss_recon:.5f} | DC loss: {loss_dc:.5f} | Reg loss: {loss_reg:.5f} | PSNR: {psnr_step:.5f}")
        
        list_of_slices = None
        if iteration % 50 == 0 and recon_step == 0:
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.altopt_results_path, f"recon_total_step_{total_steps}", axis_names = ["coronal","saggital","axial"], dir_name="recons_per_iter")

        ### Save reconstruction at minimum dc loss
        if loss_dc==self.alt_opt_meters_per_example["recon_loss_dc"].best_val:
            self.reconstruction_min_reconDC_loss = recon
            self.motion_params_min_reconDC_loss = pred_motion_params

        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,-1]]
        pred_mp_streched, list_of_track_dc_losses_aligned, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=list_of_track_dc_losses)
            
        if recon_step == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind=-1, fig_name='motion_pred_params.png')
            if iteration % 30 == 0:
                self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')
                self.save_figure_motion_parameters_dc_loss(iteration, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}_motParams.png", plot_dc_losses=False)


    def evaluate_after_motion_step(self, pred_motion_params, traj, loss_motion, iteration, motion_step, total_steps):

        self.alt_opt_meters_per_example["motion_loss"].update(loss_motion, total_steps)
        self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].update(loss_motion, total_steps)

        pred_mp_streched, _, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
        L1_motion_parameters = torch.sum(torch.abs(pred_mp_streched.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        self.alt_opt_meters_per_example["L1_gt_motion_parameters"].update(L1_motion_parameters, total_steps)
        
        self.pred_motion_params_over_epochs = torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1)

        if iteration % 10 == 0:
            pred_mp_streched_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
            L1_motion_parameters_aligned = torch.sum(torch.abs(pred_mp_streched_aligned.cpu()-self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        else:
            L1_motion_parameters_aligned = 0


        ### Log the data
        logging.info(f"total step {total_steps}, iteration {iteration}, motion step {motion_step} -- motion loss: {loss_motion:.5f} | L1(gt_motion_params): {L1_motion_parameters:.5f} | L1(gt_motion_params_aligned): {L1_motion_parameters_aligned:.5f}")


    def evaluate_after_alt_opt(self, traj):

        torch.save(self.reconstruction_min_reconDC_loss.cpu(), self.args.altopt_results_path+"/reconstruction_min_reconDC_loss.pt")
        pickle.dump(self.alt_opt_meters_per_example, open( os.path.join(self.args.altopt_results_path, 'alt_opt_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        # load the best reconstruction according to DC loss
        psnr = PSNR_torch(complex_abs(self.reconstruction_min_reconDC_loss), self.ref_img3D)
        logging.info(f"PSNR of the best reconstruction according to min recon DC loss: {psnr}")

        best_step = self.alt_opt_meters_per_example["recon_loss_dc"].best_epoch
        best_reconDC_loss = self.alt_opt_meters_per_example["recon_loss_dc"].best_val
        ind = self.alt_opt_meters_per_example["recon_loss_dc"].best_count

        list_of_slices = None
        save_slice_images_from_volume(self.reconstruction_min_reconDC_loss[0].cpu(), list_of_slices, self.args.altopt_results_path, "recon_altOpt_min_reconDC_loss", axis_names = ["coronal","saggital","axial"])


        # Load motion parameters from the best reconstruction according to DC loss
        pred_motion_params_final = self.motion_params_min_reconDC_loss.cpu().numpy()

        list_of_track_dc_losses = [self.track_dc_losses_per_state_norm_per_state[:,ind]]
        pred_mp_streched, list_of_track_dc_losses_aligned, reduce_indicator = expand_mps_to_kspline_resolution(torch.from_numpy(pred_motion_params_final).cuda(self.args.gpu), traj, list_of_track_dc_losses=list_of_track_dc_losses)

        L1_motion_parameters_final = torch.sum(torch.abs(pred_mp_streched.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        logging.info(f"L1(gt_motion_params) of the best reconstruction according to min recon DC loss: {L1_motion_parameters_final}")

        self.final_result_dict["best_reconDC_loss"] = best_reconDC_loss
        self.final_result_dict["altopt_best_reconDC_step"] = best_step 
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,ind].cpu().numpy()
        self.final_result_dict["L1_motion_parameters"] = L1_motion_parameters_final
        self.final_result_dict["PSNR_ref_vs_recon_altopt"] = psnr
        self.final_result_dict["pred_motion_params_final"] = pred_motion_params_final
        self.final_result_dict["gt_motion_params"] = self.gt_motion_params.cpu().numpy()
        self.final_result_dict["traj"] = traj
        self.final_result_dict["gt_traj"] = self.gt_traj

        
        pred_motion_params_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
        L1_motion_parameters_aligned = torch.sum(torch.abs(pred_motion_params_aligned.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
        self.final_result_dict["L1_motion_parameters_aligned"] = L1_motion_parameters_aligned
        logging.info(f"L1(gt_motion_params_aligned) of the best reconstruction according to min recon DC loss: {L1_motion_parameters_aligned}")

        pickle.dump(self.final_result_dict, open( os.path.join(self.args.altopt_results_path, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        self.save_figure_motion_parameters_dc_loss(best_step, pred_mp_streched.cpu().numpy(), self.gt_motion_params.cpu().numpy(), list_of_track_dc_losses_aligned[0], dc_loss_ind=ind, fig_name=f'motion_pred_params_min_reconDC_loss_totstep_{best_step}.png', )

        DC_losses = self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].val
        steps_recon_and_motion = self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].epochs
        steps_recon = self.alt_opt_meters_per_example["recon_loss_dc"].epochs
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(steps_recon_and_motion,DC_losses)
        plt.xlabel('Steps')
        plt.ylabel('DC loss')
        plt.title('DC loss over steps')
        plt.subplot(1,2,2)
        for i in range(self.track_dc_losses_per_state_norm_per_state.shape[0]):
            plt.plot(steps_recon,self.track_dc_losses_per_state_norm_per_state[i,:].cpu().numpy(), alpha=0.1, color='blue')
        plt.xlabel('Steps')
        plt.ylabel('DC loss per state')
        plt.title('DC loss per state over steps')
        plt.savefig(self.args.altopt_results_path+"/dc_loss_convergence.png")

    def save_figure_motion_parameters_dc_loss(self, iteration, pred_mp, gt_mp, dc_losses_1, dc_loss_ind=-1, fig_name=None, plot_dc_losses = True):

        save_dir = os.path.join(self.args.altopt_results_path, "motion_param_figures")
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



    