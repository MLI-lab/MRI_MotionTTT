import torch
import logging
import pickle
import os
import numpy as np
import scipy.signal as signal

from functions.helpers.meters import TrackMeter
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import  motion_alignment, DC_loss_thresholding, expand_mps_to_kspline_resolution, compute_discretization_error
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT, get_maxKsp_shot


from functions.helpers.helpers_img_metrics import PSNR_torch
from torch.autograd import Variable
import ptwt, pywt

class AltOptModuleBase():

    def __init__(
            self,
            args,
            ) -> None:
        
        self.args = args
        self.final_result_dict = {}

    def run_alt_opt(self):
        for name,meter in zip(self.alt_opt_meters_per_example.keys(), self.alt_opt_meters_per_example.values()):
            meter.reset()

        # # Load data
        traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D = self.load_data_init_motion()

        ###############
        # Init Reconstruction Volume
        mse = torch.nn.MSELoss()
        recon = Variable(torch.zeros(masked_corrupted_kspace3D.shape[1:])).cuda(self.args.gpu)
        recon.data.uniform_(0,1)

        if self.args.altopt_motion_estimation_only:
            if self.args.alt_opt_vivo:
                raise NotImplementedError("Motion estimation only not implemented yet for invivo.")
            logging.info("Motion estimation only. Set recon to reference image")
            recon = self.ref_img3D.clone()

        optimizer_recon = self.init_optimizer(recon, self.args.altopt_optimizer_recon, self.args.altopt_lr_recon)
        
        ###############
        # Init motion parameter estimation
        pred_motion_params, traj_new = self.init_pred_motion_params(traj)
        if traj_new is not None:
            traj = traj_new
        self.pred_motion_params_over_epochs = torch.zeros(self.args.Ns, 6, 1)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.altopt_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu).type(torch.float32)        

        if not self.args.alt_opt_vivo:
            # Expand pred_motion_params to k-space line resolution
            pred_mp_streched, _, reduce_indicator = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
            discretization_error = compute_discretization_error(pred_motion_params, traj, self.gt_motion_params)
            logging.info(f"Expand pred_motion_params to match k-space line resolution. Num states before expansion: {pred_motion_params.shape[0]} and after expansion: {pred_mp_streched.shape[0]}")
            logging.info(f"L1 loss of (expanded) motion parameters: {torch.sum(torch.abs(pred_mp_streched-self.gt_motion_params))/torch.prod(torch.tensor(self.gt_motion_params.shape))} vs. discretization error: {discretization_error}")
            self.final_result_dict["discretization_error_before_dcTh"] = discretization_error
        
        # # Apply DC loss thresholding before recon only
        if self.args.altopt_dc_thresholding and self.args.altopt_recon_only and (self.args.alt_opt_on_TTTexp or self.args.alt_opt_on_alt_opt_exp):
            num_lines_before_th = torch.sum(masks2D_all_states)
            pred_motion_params, traj, masks2D_all_states, masked_corrupted_kspace3D, energy_discarded, num_lines_discarded = self.run_DC_loss_thresholding(pred_motion_params, traj, masks2D_all_states, masked_corrupted_kspace3D)

            logging.info(f"During thresholding percentage of k-space energy discarded: {energy_discarded} and num lines discarded: {num_lines_discarded} (out of {num_lines_before_th}, i.e. {num_lines_discarded/num_lines_before_th} percent).")
            self.final_result_dict['percentage_energy_discarded'] = energy_discarded
            self.final_result_dict['num_lines_discarded'] = num_lines_discarded
            
        # # Align motion parameters with ground truth motion parameters before recon only
        if not self.args.alt_opt_vivo:
            if self.args.altopt_align_motParams and self.args.altopt_recon_only and (self.args.alt_opt_on_TTTexp or self.args.alt_opt_on_alt_opt_exp):
                logging.info("Aligning motion parameters with ground truth motion parameters.")

                # Expand pred_motion_params to k-space line resolution
                pred_mp_streched, _, reduce_indicator = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
                logging.info(f"Expand pred_motion_params to match k-space line resolution. Num states before expansion: {pred_motion_params.shape[0]} and after expansion: {pred_mp_streched.shape[0]}")
                
                # Align pred_motion_params with gt_motion_params
                pred_mp_streched_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
                logging.info(f"L1 loss of aligned motion parameters: {torch.sum(torch.abs(pred_mp_streched_aligned.cpu()-self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))}")
                
                # Reduce aligned pred_motion_params to original resolution
                reduce_indicator_shifted = torch.zeros_like(reduce_indicator)
                reduce_indicator_shifted[0] = reduce_indicator[0]-1
                reduce_indicator_shifted[1:] = reduce_indicator[:-1]
                difference = reduce_indicator - reduce_indicator_shifted
                reduce_indices = torch.where(difference != 0)[0]
                pred_motion_params = pred_mp_streched_aligned[reduce_indices]

        optimizer_motion = self.init_optimizer(pred_motion_params, self.args.altopt_optimizer_motion, self.args.altopt_lr_motion)
        
        # # Exclude shot with max k-space energy from motion estimation
        if not self.args.altopt_recon_only:
            shot_ind_maxksp = get_maxKsp_shot(masked_corrupted_kspace3D, traj, self.args.fix_mot_maxksp_shot)

            if self.args.fix_mot_maxksp_shot and self.gt_motion_params is not None:
                if shot_ind_maxksp != 0 and self.args.motionTraj_simMot != "uniform_interShot_event_model":
                    raise ValueError("fix_mot_maxksp_shot with shot_ind_maxksp not equal 0 is only supported for uniform_interShot_event_model.")
                
                # Set pred_motion_params for shot with max k-space energy to gt_motion_params
                pred_motion_params[shot_ind_maxksp] = self.gt_motion_params[shot_ind_maxksp]
                logging.info(f"Set pred_motion_params for shot with max k-space energy to gt_motion_params: {pred_motion_params[shot_ind_maxksp].cpu().numpy()}")


        logging.info(f"""Starting Alt Opt with {self.args.altopt_steps_total} total steps, 
                     {self.args.altopt_steps_recon} recon steps with lr {self.args.altopt_lr_recon:.1e}, lambda {self.args.altopt_lam_recon:.1e} and optimizer {self.args.altopt_optimizer_recon},
                     and {self.args.altopt_steps_motion} motion est steps with lr {self.args.altopt_lr_motion:.1e} and optimizer {self.args.altopt_optimizer_motion}.
                     Recon only is {self.args.altopt_recon_only} and motion est only is {self.args.altopt_motion_estimation_only}.""")
        
        #masked_corrupted_kspace3D,_,_ = normalize_separate_over_ch_3D(torch.moveaxis(masked_corrupted_kspace3D,-1,1), eps=1e-11)
        #masked_corrupted_kspace3D = torch.moveaxis(masked_corrupted_kspace3D,1,-1)

        # Normalize k-space to 0 mean and std 1. This hopefully allows reconstructing
        # data that was stored on a different scale.
        #masked_corrupted_kspace3D,_,_ = normalize_instance(masked_corrupted_kspace3D,eps=1e-11)

        new_phase = False        

        ksp_shape = masked_corrupted_kspace3D.shape

        total_steps = 0
        for iteration in range(self.args.altopt_steps_total):
            
            if not self.args.altopt_motion_estimation_only:
                recon.requires_grad = True
                pred_motion_params.requires_grad = False
                for recon_step in range(self.args.altopt_steps_recon):
                    
                    self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if total_steps > 0 else torch.zeros(self.args.Ns,1)
                    self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if total_steps > 0 else torch.zeros(self.args.Ns,1)
            
                    total_steps += 1
                    optimizer_recon.zero_grad()
                    
                    # Step 1: Apply forward model
                    # a. Expand opeartor:
                    recon_coil = complex_mul(recon.unsqueeze(0), smaps3D)
                    # b. Apply Mask:
                    recon_kspace3d_coil = fft2c_ndim(recon_coil, 3)

                    recon_kspace3d_coil = motion_corruption_NUFFT(recon_kspace3d_coil, recon_coil, pred_motion_params, traj, 
                                                                  weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size)
                    
                    # Step 2: Calculating the loss and backward
                    # a. take wavelet of reconstruction
                    coefficient = ptwt.wavedec3(recon, pywt.Wavelet("haar"),level=1)[0]
                    # b. Calculating the loss and backward
                    #loss_dc = torch.sum(torch.abs(recon_kspace3d_coil - masked_corrupted_kspace3D)**2) / torch.sum(torch.abs(masked_corrupted_kspace3D)**2)
                    loss_dc = mse( recon_kspace3d_coil , masked_corrupted_kspace3D )
                    loss_reg = self.args.altopt_lam_recon*torch.norm(coefficient,p=1)
                    loss_recon = loss_dc + loss_reg

                    loss_recon.backward()
                    optimizer_recon.step()

                    self.update_dc_losses_per_state(recon_kspace3d_coil.detach(), masked_corrupted_kspace3D, masks2D_all_states)
                    
                    recon_for_eval = recon.detach()*binary_background_mask
                    self.evaluate_after_recon_step(recon_for_eval, pred_motion_params.detach(), traj, loss_recon.item(), loss_reg.item(), loss_dc.item(), iteration, recon_step, total_steps)
                optimizer_recon.zero_grad()

            if not self.args.altopt_recon_only:

                recon.requires_grad = False
                pred_motion_params.requires_grad = True

                # Step 1: Apply forward model
                # a. Expand opeartor:
                recon_coil = complex_mul(recon.unsqueeze(0), smaps3D)
                # b. Apply Mask:
                recon_kspace3d_coil = fft2c_ndim(recon_coil, 3)
                for motion_step in range(self.args.altopt_steps_motion):
                    total_steps += 1
                    optimizer_motion.zero_grad()
                    
                    recon_kspace3d_coil_corrupted = motion_corruption_NUFFT(recon_kspace3d_coil, recon_coil, pred_motion_params, traj, 
                                                                            weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size)
                    #recon_kspace3d_coil_corrupted,_,_ = normalize_instance(recon_kspace3d_coil_corrupted,eps=1e-11)

                    #loss_motion = torch.sum(torch.abs(recon_kspace3d_coil - masked_corrupted_kspace3D)**2) / torch.sum(torch.abs(masked_corrupted_kspace3D)**2)
                    loss_motion = mse( recon_kspace3d_coil_corrupted , masked_corrupted_kspace3D )
                    loss_motion.backward()
                    optimizer_motion.step()

                    self.evaluate_after_motion_step(pred_motion_params.detach(), traj, loss_motion.item(), iteration, motion_step, total_steps)

                optimizer_motion.zero_grad()

        self.evaluate_after_alt_opt(traj)

    def init_optimizer(self, param, optimizer_type, lr):
        '''
        Run this to define the optimizer for recon or motion estimation.
        '''
        
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam([param], lr=lr)
            logging.info(f"Init Adam optimizer with lr {lr}")
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD([param], lr=lr)
            logging.info(f"Init SGD optimizer with lr {lr}")
        else:
            raise ValueError("Unknown optimizer")
        
        return optimizer
    
    def init_pred_motion_params(self, traj_old):
        '''
        This function can 
        - set pred_motion_params to the ground truth motion parameters
        - load pred_motion_params from a previous alt opt run
        - load pred_motion_params from a previous motion TTT run
        '''
        load_counter = 0

        if self.args.altopt_recon_only and self.args.altopt_recon_only_with_motionKnowledge and not self.args.alt_opt_vivo:
            if self.args.alt_opt_vivo:
                raise NotImplementedError("Recon only with motion knowledge not implemented yet for invivo.")
            logging.info(f"Using ground truth motion parameters for reconstruction with altopt_recon_only_with_motionKnowledge_discretized {self.args.altopt_recon_only_with_motionKnowledge_discretized}.")

            if self.args.altopt_recon_only_with_motionKnowledge_discretized:
                traj = traj_old

                # Get resoltion of pred_motion_params
                if self.loaded_pred_motion_params is None:
                    logging.info("Initialize pred_motion_params with all zeros.")
                    pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
                else:
                    logging.info("Initialize pred_motion_params with loaded_pred_motion_params.")
                    pred_motion_params = self.loaded_pred_motion_params.clone()

                gt_motion_params_discrete = torch.zeros_like(pred_motion_params)
                assert len(traj[0]) == pred_motion_params.shape[0]
                running_ind = 0
                for i in range(len(traj[0])):
                    gt_motion_params_discrete[i] = torch.mean(self.gt_motion_params[running_ind:running_ind+len(traj[0][i])], dim=0)
                    running_ind += len(traj[0][i])

                pred_motion_params = gt_motion_params_discrete

                # Reduce the number of motion states by combining motion states with the same motion parameters
                pred_motion_params_combined = pred_motion_params[0:1,:]
                traj_combined = ([traj[0][0]], [traj[1][0]])
                for i in range(1, pred_motion_params.shape[0]):
                    if torch.sum(torch.abs(pred_motion_params[i]-pred_motion_params[i-1])) > 0:
                        pred_motion_params_combined = torch.cat((pred_motion_params_combined, pred_motion_params[i:i+1,:]), dim=0)
                        traj_combined[0].append(traj[0][i]) 
                        traj_combined[1].append(traj[1][i])
                    else:
                        traj_combined[0][-1] = np.concatenate((traj_combined[0][-1], traj[0][i]), axis=0)
                        traj_combined[1][-1] = np.concatenate((traj_combined[1][-1], traj[1][i]), axis=0)

                traj = traj_combined
                pred_motion_params = pred_motion_params_combined

            elif self.args.altopt_recon_only_with_motionKnowledge_remove_intraMotion:
                pred_motion_params = self.gt_motion_params[0:1,:]
                traj = ([traj_old[0][0]], [traj_old[1][0]])
                i=len(traj[0][0])
                j=1
                while i < self.gt_motion_params.shape[0]:
                    if len(traj_old[0][j]) > len(traj_old[0][0])-20:
                        pred_motion_params = torch.cat((pred_motion_params, self.gt_motion_params[i:i+1,:]), dim=0)
                        traj[0].append(traj_old[0][j]) 
                        traj[1].append(traj_old[1][j])
                    i += len(traj_old[0][j])
                    j += 1

                # Update ground truth parameters
                self.gt_motion_params, _, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
                self.gt_traj = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])
                for i in torch.arange(1,len(traj[0])):
                    # For each shot expand the traj to per line resolution
                    self.gt_traj[0].extend([np.array([k]) for k in traj[0][i]])
                    self.gt_traj[1].extend([np.array([k]) for k in traj[1][i]])

                # Reduce the number of motion states by combining motion states with the same motion parameters
                pred_motion_params_combined = pred_motion_params[0:1,:]
                traj_combined = ([traj[0][0]], [traj[1][0]])
                for i in range(1, pred_motion_params.shape[0]):
                    if torch.sum(torch.abs(pred_motion_params[i]-pred_motion_params[i-1])) > 0:
                        pred_motion_params_combined = torch.cat((pred_motion_params_combined, pred_motion_params[i:i+1,:]), dim=0)
                        traj_combined[0].append(traj[0][i]) 
                        traj_combined[1].append(traj[1][i])
                    else:
                        traj_combined[0][-1] = np.concatenate((traj_combined[0][-1], traj[0][i]), axis=0)
                        traj_combined[1][-1] = np.concatenate((traj_combined[1][-1], traj[1][i]), axis=0)

                traj = traj_combined
                pred_motion_params = pred_motion_params_combined

            else:
                # Reduce the number of motion states by combining motion states with the same motion parameters
                pred_motion_params = self.gt_motion_params[0:1,:]
                traj = ([self.gt_traj[0][0]], [self.gt_traj[1][0]])
                for i in range(1, self.gt_motion_params.shape[0]):
                    if torch.sum(torch.abs(self.gt_motion_params[i]-self.gt_motion_params[i-1])) > 0:
                        pred_motion_params = torch.cat((pred_motion_params, self.gt_motion_params[i:i+1,:]), dim=0)
                        traj[0].append(self.gt_traj[0][i]) 
                        traj[1].append(self.gt_traj[1][i])
                    else:
                        traj[0][-1] = np.concatenate((traj[0][-1], self.gt_traj[0][i]), axis=0)
                        traj[1][-1] = np.concatenate((traj[1][-1], self.gt_traj[1][i]), axis=0)

            load_counter += 1
            #traj = self.gt_traj
            self.args.Ns = pred_motion_params.shape[0]
            logging.info(f"Number of motion states after combining states with the same motion parameters: {pred_motion_params.shape[0]}")

        if self.args.altopt_recon_only and self.args.alt_opt_on_TTTexp:
            motion_TTT_results_path = os.path.join(self.args.TTT_results_path_numerical, f"phase{self.args.alt_opt_on_TTT_load_from_phase}/final_result_dict.pkl") # phase loading !!!
            logging.info(f"Load pred_motion_paramsm, traj, gt_motion_params from motion TTT phase {self.args.alt_opt_on_TTT_load_from_phase} from {motion_TTT_results_path}")
            with open(motion_TTT_results_path,'rb') as fn:
                final_results_dict_TTT = pickle.load(fn)
            pred_motion_params = torch.from_numpy(final_results_dict_TTT['pred_motion_params_final']).cuda(self.args.gpu)
            # gt_motion_params should not be loaded but should have already be generated correctly as it also determines the generation of the corrupted k-space
            # if final_results_dict_TTT['gt_motion_params'] is not None:
            #     self.gt_motion_params = torch.from_numpy(final_results_dict_TTT['gt_motion_params']).cuda(self.args.gpu)
            # else:
            #     self.gt_motion_params = None
            traj = final_results_dict_TTT['traj']
            self.args.Ns = pred_motion_params.shape[0]

            load_counter += 1
        
        if self.args.altopt_recon_only and self.args.alt_opt_on_alt_opt_exp:
            altopt_results_path = os.path.join(self.args.altopt_load_path, 'final_result_dict.pkl')
            logging.info(f"Load pred_motion_params, traj, gt_motion_params from alt opt from {altopt_results_path}")
            with open(altopt_results_path,'rb') as fn:
                final_results_dict_altopt = pickle.load(fn)
            pred_motion_params = torch.from_numpy(final_results_dict_altopt['pred_motion_params_final']).cuda(self.args.gpu)
            # gt_motion_params should not be loaded but should have already be generated correctly as it also determines the generation of the corrupted k-space
            # if final_results_dict_altopt['gt_motion_params'] is not None:
            #     self.gt_motion_params = torch.from_numpy(final_results_dict_altopt['gt_motion_params']).cuda(self.args.gpu)
            # else:
            #     self.gt_motion_params = None
            traj = final_results_dict_altopt['traj']
            self.args.Ns = pred_motion_params.shape[0]

            load_counter += 1

        if load_counter > 1:
            raise ValueError("Multiple ways to load pred_motion_params.")
        elif load_counter == 0:
            if self.loaded_pred_motion_params is None:
                logging.info("Initialize pred_motion_params with all zeros.")
                pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
            else:
                logging.info("Initialize pred_motion_params with loaded_pred_motion_params.")
                pred_motion_params = self.loaded_pred_motion_params.clone()
            traj = None

        return pred_motion_params, traj

    def run_DC_loss_thresholding(self, pred_motion_params, traj, masks2D_all_states, masked_corrupted_kspace3D):
        
        # # Load dc_loss_per_state_norm_per_state
        
        if self.args.alt_opt_on_TTTexp:

            # # Load dc_loss_per_state_norm_per_state
            motion_TTT_results_path = os.path.join(self.args.TTT_results_path_numerical, f"phase{self.args.alt_opt_on_TTT_load_from_phase}/final_result_dict.pkl") # phase loading !!!
            logging.info(f"Load final_result_dict from motion TTT phase {self.args.alt_opt_on_TTT_load_from_phase} from {motion_TTT_results_path}")
            with open(motion_TTT_results_path,'rb') as fn:
                final_results_dict_TTT = pickle.load(fn)
            dc_loss_per_state_norm_per_state = final_results_dict_TTT['dc_losses_per_state_norm_per_state_min_reconDC_loss']

        elif self.args.alt_opt_on_alt_opt_exp:
            # # Load dc_loss_per_state_norm_per_state
            altopt_results_path = os.path.join(self.args.altopt_load_path, 'final_result_dict.pkl')
            logging.info(f"Load final_result_dict from alt opt from {altopt_results_path}")
            with open(altopt_results_path,'rb') as fn:
                final_results_dict_altopt = pickle.load(fn)
            dc_loss_per_state_norm_per_state = final_results_dict_altopt['dc_losses_per_state_norm_per_state_min_reconDC_loss']

        # # Apply peak or hard thresholding
            
        masked_corrupted_kspace3D_new, gt_traj_new, traj_new, gt_motion_params_new, pred_motion_params_new, masks2D_all_states_new, Ns, dc_th_states_ind, discretization_error = DC_loss_thresholding(dc_loss_per_state_norm_per_state, self.args.altopt_dc_threshold, 
                                                                                                                             self.gt_traj, traj, self.gt_motion_params, pred_motion_params, 
                                                                                                                             masks2D_all_states, masked_corrupted_kspace3D)
        

        # _, _, _, self.gt_mp_ksp_reso, _, _, _ = DC_loss_thresholding(dc_loss_per_state_norm_per_state, self.args.altopt_dc_threshold, 
        #                                                                                                                      self.gt_traj, traj, self.gt_mp_ksp_reso, pred_motion_params, 
        #                                                                                                                   masks2D_all_states, masked_corrupted_kspace3D)           
        
        # # Compute the fraction of k-space energy that is discarded and the number of k-spaces lines that are discarded
        energy_discarded = 0
        num_lines_discarded = 0

        if pred_motion_params.shape[0] != pred_motion_params_new.shape[0]:
            dc_th_states_ind_to_exclude = np.setdiff1d(np.arange(0,pred_motion_params.shape[0]), dc_th_states_ind)

            for state in dc_th_states_ind_to_exclude:
                energy_discarded += torch.sum(torch.abs(masked_corrupted_kspace3D*masks2D_all_states[state])) 
                num_lines_discarded += torch.sum(masks2D_all_states[state])

            energy_discarded = energy_discarded / torch.sum(torch.abs(masked_corrupted_kspace3D))

        self.final_result_dict["discretization_error_after_dcTh"] = discretization_error
        self.args.Ns = Ns
        self.gt_motion_params = gt_motion_params_new
        self.gt_traj = gt_traj_new
        masked_corrupted_kspace3D = masked_corrupted_kspace3D_new
        traj = traj_new
        pred_motion_params = pred_motion_params_new
        masks2D_all_states = masks2D_all_states_new

        return pred_motion_params, traj, masks2D_all_states, masked_corrupted_kspace3D, energy_discarded, num_lines_discarded
    
    def update_dc_losses_per_state(self, recon_kspace3D, masked_corrupted_kspace3D, masks2D_all_states):
        '''
        This function computes the data consistency loss for each stat, which can be 
        used for DC loss thresholding after alt opt.
        '''
        for i in range(masks2D_all_states.shape[0]):
            mask3D_tmp = masks2D_all_states[i]
            dc_loss_norm_per_state = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
            dc_loss_norm_all_states = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            self.track_dc_losses_per_state_norm_per_state[i,-1] = dc_loss_norm_per_state
            self.track_dc_losses_per_state_norm_all_states[i,-1] = dc_loss_norm_all_states
            
    
    def load_data_init_motion(self):
        '''
        This function must be implemented in the derived class.
        It should load data from a single volume (e.g. from CC359 or invivo).
        The following quantities HAVE to be returned to run TTT_step (on gpu):
            - traj: Sampling trajectory
            - smaps3D: Sensitivity maps, torch.Tensor (coils,x,y,z,2)
            - binary_background_mask: Binary mask torch.Tensor (x,y,z,1)
            - masked_corruped_kspace3D: Motion-corrupted k-space torch.Tensor (coils,x,y,z,2)
            - mask3D: Mask torch.Tensor (1,x,y,z,1)
        Optional quantities:
            - gt_motion_params: Ground truth motion parameters, torch.Tensor (Ns,6)
            - ref_img3D: Reference image, torch.Tensor (x,y,z,2)
        '''
        pass
                    
    def init_altopt_meters(self):
        '''
        This function must be implemented in the derived class.
        It should initialize altopt_meters_per_example a dictionary containing 
        TrackMeters for each quantity that should be tracked during TTT.
        '''
        pass

    def evaluate_before_altopt(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance before altopt and must be called by
        load_data_init_motion().
        '''
        pass

    def evaluate_after_recon_step(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after one recon step of alt opt.
        '''
        pass

    def evaluate_after_motion_step(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after one motion estimation step of alt opt.
        '''
        pass

    def evaluate_after_alt_opt(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after altopt.
        '''
        pass
