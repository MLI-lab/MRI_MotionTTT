import torch
import logging
import pickle
import os
import numpy as np

from functions.utils.helpers.helpers_math import complex_mul, fft2c_ndim
from functions.utils.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories
from functions.utils.motion_simulation.motion_helpers import get_maxKsp_shot, motion_alignment, DC_loss_thresholding, expand_mps_to_kspline_resolution, compute_discretization_error
from functions.utils.motion_simulation.motion_forward_backward_models import motion_corruption_NUFFT

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
            recon = self.ref_img3D_complex.clone()

        optimizer_recon = self.init_optimizer(recon, self.args.altopt_optimizer_recon, self.args.altopt_lr_recon)
        
        ###############
        # Init motion parameter estimation
        pred_motion_params, traj_new = self.init_pred_motion_params()
        if traj_new is not None:
            traj = traj_new
        self.pred_motion_params_over_epochs = torch.zeros(self.args.Ns, 6, 1)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.altopt_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu).type(torch.float32) 

        optimizer_motion = self.init_optimizer(pred_motion_params, self.args.altopt_optimizer_motion, self.args.altopt_lr_motion)
        
        # # Exclude shot with max k-space energy from motion estimation
        if not self.args.altopt_recon_only:
            # get_maxKsp_shot returns None if fix_mot_maxksp_shot is False
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
                                                                            weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size,
                                                                            shot_ind_maxksp=shot_ind_maxksp)

                    loss_motion = mse( recon_kspace3d_coil_corrupted , masked_corrupted_kspace3D )
                    loss_motion.backward()
                    optimizer_motion.step()

                    self.evaluate_after_motion_step(pred_motion_params.detach(), traj, loss_motion.item(), iteration, motion_step, total_steps)

                optimizer_motion.zero_grad()
            #  Thresholding
            if self.args.is_altopt_threshold:
                if iteration == 0:
                    last_loss_recon = loss_recon.item()
                else:
                    if ((np.log(last_loss_recon-loss_recon.item())<self.args.altopt_threshold) and ((last_loss_recon-loss_recon.item())>0)) and (iteration>30):
                        break
                    else:
                        last_loss_recon = loss_recon.item()
                        
        self.evaluate_after_alt_opt(traj)

    
    def init_pred_motion_params(self):
        '''
        This function can 
        - set pred_motion_params to the ground truth motion parameters if altopt_recon_only
        - else set pred_motion_params to all zeros
        '''

        if self.args.altopt_recon_only:
            if self.args.alt_opt_vivo:
                raise NotImplementedError("Recon only with motion knowledge not implemented yet for invivo.")
            
            logging.info(f"Initialize pred_motion_params with gt_motion_params because altopt_recon_only is {self.args.altopt_recon_only}.")
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

            self.args.Ns = pred_motion_params.shape[0]
            logging.info(f"Number of motion states after combining states with the same motion parameters: {pred_motion_params.shape[0]}")

        else:
            logging.info("Initialize pred_motion_params with all zeros.")
            pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
            traj = None

        return pred_motion_params, traj

    def update_dc_losses_per_state(self, recon_kspace3D, masked_corrupted_kspace3D, masks2D_all_states):
        '''
        This function computes the data consistency loss for each state, which can be 
        used for DC loss thresholding after alt opt.
        '''
        for i in range(masks2D_all_states.shape[0]):
            mask3D_tmp = masks2D_all_states[i]
            dc_loss_norm_per_state = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
            dc_loss_norm_all_states = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            self.track_dc_losses_per_state_norm_per_state[i,-1] = dc_loss_norm_per_state
            self.track_dc_losses_per_state_norm_all_states[i,-1] = dc_loss_norm_all_states

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
            
    
    def load_data_init_motion(self):
        '''
        This function must be implemented in the derived class.
        It should load data from a single volume (e.g. from CC359 or invivo).
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
