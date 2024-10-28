import torch
import logging
from torch.autograd import Variable
import ptwt, pywt
import numpy as np
import pickle

from functions.utils.helpers.helpers_math import complex_mul, fft2c_ndim
from functions.utils.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories
from functions.utils.motion_simulation.motion_helpers import motion_alignment, DC_loss_thresholding, expand_mps_to_kspline_resolution
from functions.utils.motion_simulation.motion_forward_backward_models import motion_corruption_NUFFT


class L1minModuleBase():

    def __init__(
            self,
            args,
            ) -> None:
        
        self.args = args
        self.final_result_dict = {}

    def run_L1min(self):
        for name,meter in zip(self.L1min_meters.keys(), self.L1min_meters.values()):
            meter.reset()

        # # Load data
        traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D = self.load_data_init_motion()

        ###############
        # Init Reconstruction Volume
        mse = torch.nn.MSELoss()
        recon = Variable(torch.zeros(masked_corrupted_kspace3D.shape[1:])).cuda(self.args.gpu)
        recon.data.uniform_(0,1)
        recon.requires_grad = True

        optimizer = self.init_optimizer(recon, self.args.L1min_optimizer, self.args.L1min_lr)

        ###############
        # Init motion parameter estimation
        pred_motion_params, traj_new = self.init_pred_motion_params()
        if traj_new is not None:
            traj = traj_new
        self.pred_motion_params_over_epochs = torch.zeros(self.args.Ns, 6, 1)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.L1min_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu).type(torch.float32)        

        # If gt motion available, log L1 motion error
        if self.gt_motion_params is not None:
            pred_mp_streched, _, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
            motion_error = torch.sum(torch.abs(pred_mp_streched.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
            logging.info(f"Initial L1 motion error (no DC loss thresholding or motion alignment): {motion_error}")
            self.final_result_dict['L1_motion_error_initial'] = motion_error

        # Apply DC loss thresholding
        if self.args.L1min_DC_loss_thresholding:
            assert self.args.L1min_mode != 'gt_motion', "DC loss thresholding not implemented for gt_motion mode."
            logging.info(f"Apply DC loss thresholding with threshold {self.args.L1min_DC_threshold}.")

            num_lines_before_th = torch.sum(masks2D_all_states)
            num_states_before_th = pred_motion_params.shape[0]
            (pred_motion_params, traj, masks2D_all_states, 
             masked_corrupted_kspace3D, energy_discarded, 
             num_lines_discarded) = self.run_DC_loss_thresholding(pred_motion_params, traj, 
                                                                  masks2D_all_states, masked_corrupted_kspace3D)

            num_states_after_th = pred_motion_params.shape[0]
            logging.info(f"During thresholding percentage of k-space energy discarded: {energy_discarded} and num lines discarded: {num_lines_discarded} (out of {num_lines_before_th}, i.e. {num_lines_discarded/num_lines_before_th} percent).")
            logging.info(f"Number of motion states before and after DC loss thresholding: {num_states_before_th} and {num_states_after_th}")
            self.final_result_dict['percentage_energy_discarded'] = energy_discarded
            self.final_result_dict['num_lines_discarded'] = num_lines_discarded

            if self.gt_motion_params is not None:
                pred_mp_streched, _, _ = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
                motion_error = torch.sum(torch.abs(pred_mp_streched.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
                logging.info(f"L1 motion error after DC loss thresholding: {motion_error}")
                self.final_result_dict['L1_motion_error_DC_th'] = motion_error

        # Align predicted motion parameters with ground truth motion parameters
        if self.gt_motion_params is not None and self.args.L1min_motion_alignment:
            logging.info("Align predicted motion parameters with ground truth motion parameters.")
            
            # Expand pred_motion_params to k-space line resolution
            pred_mp_streched, _, reduce_indicator = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None)
            
            # Align pred_motion_params with gt_motion_params
            pred_mp_streched_aligned = motion_alignment(pred_mp_streched, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
            motion_error = torch.sum(torch.abs(pred_mp_streched_aligned.cpu() - self.gt_motion_params.cpu()))/torch.prod(torch.tensor(self.gt_motion_params.shape))
            logging.info(f"L1 motion error after motion alignment: {motion_error}")
            self.final_result_dict['L1_motion_error_motion_alignment'] = motion_error
            
            # Reduce aligned pred_motion_params to original resolution
            reduce_indicator_shifted = torch.zeros_like(reduce_indicator)
            reduce_indicator_shifted[0] = reduce_indicator[0]-1
            reduce_indicator_shifted[1:] = reduce_indicator[:-1]
            difference = reduce_indicator - reduce_indicator_shifted
            reduce_indices = torch.where(difference != 0)[0]
            pred_motion_params = pred_mp_streched_aligned[reduce_indices]

        ###############
        # Run L1-min
            
        logging.info(f"Starting L1-minimization with {self.args.L1min_num_steps} steps, lr {self.args.L1min_lr:.1e}, lambda {self.args.L1min_lambda:.1e} and optimizer {self.args.L1min_optimizer}.")
        
        pred_motion_params.requires_grad = False
        for iteration in range(self.args.L1min_num_steps):

            self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            
            optimizer.zero_grad()
            
            # Step 1: Apply forward model
            recon_coil = complex_mul(recon.unsqueeze(0), smaps3D)
            recon_kspace3d_coil = fft2c_ndim(recon_coil, 3)
            recon_kspace3d_coil = motion_corruption_NUFFT(recon_kspace3d_coil, recon_coil, pred_motion_params, traj, 
                                                            weight_rot=True, args=self.args, max_coil_size=self.args.L1min_nufft_max_coil_size)
            
            # Step 2: Calculating the loss and backward
            # a. take wavelet of reconstruction
            coefficient = ptwt.wavedec3(recon, pywt.Wavelet("haar"),level=1)[0]
            # b. Calculating the loss and backward
            loss_dc = mse( recon_kspace3d_coil , masked_corrupted_kspace3D )
            loss_reg = self.args.L1min_lambda*torch.norm(coefficient,p=1)
            loss_recon = loss_dc + loss_reg

            loss_recon.backward()
            optimizer.step()

            self.update_dc_losses_per_state(recon_kspace3d_coil.detach(), masked_corrupted_kspace3D, masks2D_all_states)
            
            recon_for_eval = recon.detach()*binary_background_mask
            self.evaluate_after_L1min_step(recon_for_eval, pred_motion_params.detach(), traj, loss_recon.item(), loss_reg.item(), loss_dc.item(), iteration)

        self.evaluate_after_L1min(traj, pred_motion_params)

    def init_pred_motion_params(self):
        '''
        This function can 
        - set pred_motion_params to the ground truth motion parameters
        - load pred_motion_params from a previous alt opt run
        - load pred_motion_params from a previous motion TTT run
        - initialize pred_motion_params with zeros
        '''

        if self.args.L1min_mode == 'gt_motion':
            logging.info(f"Using ground truth motion parameters for reconstruction with L1-min.")

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

        elif self.args.L1min_mode == 'pred_mot_motionTTT':
    
            logging.info(f"Load pred_motion_params and traj from motion TTT phase {self.args.L1min_on_TTT_load_from_phase} from {self.args.L1min_load_path}")
            with open(self.args.L1min_load_path,'rb') as fn:
                final_results_dict_TTT = pickle.load(fn)
            pred_motion_params = torch.from_numpy(final_results_dict_TTT['pred_motion_params_final']).cuda(self.args.gpu)
            traj = final_results_dict_TTT['traj']
            self.args.Ns = pred_motion_params.shape[0]

        elif self.args.L1min_mode == 'pred_mot_altopt':
            logging.info(f"Load pred_motion_params and traj from alt opt from {self.args.L1min_load_path}")
            with open(self.args.L1min_load_path,'rb') as fn:
                final_results_dict_altopt = pickle.load(fn)
            pred_motion_params = torch.from_numpy(final_results_dict_altopt['pred_motion_params_final']).cuda(self.args.gpu)
            traj = final_results_dict_altopt['traj']
            self.args.Ns = pred_motion_params.shape[0]

        elif self.args.L1min_mode == 'noMoCo':
            logging.info("Initialize pred_motion_params with all zeros.")
            pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
            traj = None
        else:
            raise Exception("L1min mode not implemented.")

        return pred_motion_params, traj

    def run_DC_loss_thresholding(self, pred_motion_params, traj, masks2D_all_states, masked_corrupted_kspace3D):
        
        # # Load dc_loss_per_state_norm_per_state
        with open(self.args.L1min_load_path,'rb') as fn:
            final_results_dict = pickle.load(fn)
        dc_loss_per_state_norm_per_state = final_results_dict['dc_losses_per_state_norm_per_state_min_reconDC_loss']
        
        # # Apply DC loss thresholding
        (masked_corrupted_kspace3D_new, gt_traj_new, traj_new, 
         gt_motion_params_new, pred_motion_params_new, masks2D_all_states_new, 
         Ns, dc_th_states_ind, _) = DC_loss_thresholding(dc_loss_per_state_norm_per_state, 
                                                                            self.args.L1min_DC_threshold, 
                                                                            self.gt_traj, traj, self.gt_motion_params, 
                                                                            pred_motion_params,  masks2D_all_states, 
                                                                            masked_corrupted_kspace3D)
         
        # # Compute the fraction of k-space energy that is discarded and the number of k-spaces lines that are discarded
        energy_discarded = 0
        num_lines_discarded = 0

        if pred_motion_params.shape[0] != pred_motion_params_new.shape[0]:
            dc_th_states_ind_to_exclude = np.setdiff1d(np.arange(0,pred_motion_params.shape[0]), dc_th_states_ind)

            for state in dc_th_states_ind_to_exclude:
                energy_discarded += torch.sum(torch.abs(masked_corrupted_kspace3D*masks2D_all_states[state])) 
                num_lines_discarded += torch.sum(masks2D_all_states[state])

            energy_discarded = energy_discarded / torch.sum(torch.abs(masked_corrupted_kspace3D))

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
        This function computes the data consistency loss for each state.
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

    def evaluate_after_L1min_step(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after one step of L1 minimization.
        '''
        pass

    def evaluate_after_L1min(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after L1-minimization.
        '''
        pass
            