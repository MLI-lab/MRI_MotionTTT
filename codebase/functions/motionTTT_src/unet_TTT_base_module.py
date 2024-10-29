import torch
import logging
import numpy as np

from functions.utils.models.helpers_model import unet_forward_all_axes
from functions.utils.motion_simulation.motion_helpers import get_maxKsp_shot
from functions.utils.motion_simulation.motion_forward_backward_models import motion_correction_NUFFT, motion_corruption_NUFFT


from functions.utils.helpers.helpers_math import complex_mul, fft2c_ndim, chunks
from functions.utils.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories

class UnetTTTModuleBase():
    def __init__(self,
                 args,
                 model) -> None:
        
        self.args = args
        self.model = model

        self.final_result_dict = {}

    def run_TTT(self):

        self.model.eval()
        # set requires_grad to False for all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            meter.reset()

        # # Load data
        traj, smaps3D, smaps3D_conj, binary_background_mask, masked_corrupted_kspace3D, mask3D = self.load_data_init_motion(evaluate_before_TTT=True)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.TTT_results_path, dir_name=f"motion_sampling_traj_phase{self.phase}", save_figures = True)).cuda(self.args.gpu)
        ###############
        # Init TTT for motion prediction
        # We assume no motion takes place during the first shot for both motionTraj_simMot equal
        # 'uniform_interShot_event_model' and 'uniform_intraShot_event_model'
        # Hence, to use fix_mot_maxksp_shot we only need to assure that the center is in the first shot
            
        if self.args.sampTraj_simMot == 'linear_cartesian' and self.args.fix_mot_maxksp_shot:
            raise ValueError("fix_mot_maxksp_shot is not supported for linear_cartesian sampling trajectory as center is not in the first shot.")
        if not self.args.center_in_first_state and self.args.fix_mot_maxksp_shot:
            raise ValueError("fix_mot_maxksp_shot is not supported for center_in_first_state=False.")
        
        logging.info("Initialize pred_motion_params with all zeros.")
        pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)

        shot_ind_maxksp = get_maxKsp_shot(masked_corrupted_kspace3D, traj, self.args.fix_mot_maxksp_shot)

        if self.args.fix_mot_maxksp_shot and self.gt_motion_params is not None:            
            # Set pred_motion_params for shot with max k-space energy to gt_motion_params
            pred_motion_params[shot_ind_maxksp] = self.gt_motion_params[shot_ind_maxksp]
            #logging.info(f"Set pred_motion_params for shot with max k-space energy to gt_motion_params: {pred_motion_params[shot_ind_maxksp].cpu().numpy()}")

        pred_motion_params.requires_grad = True

        optimizer = self.init_optimizer(pred_motion_params, self.args.lr_TTT)

        logging.info(f"fix_mot_maxksp_shot={self.args.fix_mot_maxksp_shot} and shot index with max k-space energy is {shot_ind_maxksp}.")
        logging.info(f"number of rot only grad steps: {self.args.TTT_num_rot_only_grad_steps}")
        logging.info(f"lr_max_decays {self.args.TTT_lr_max_decays}, lr_decay_factor {self.args.TTT_lr_decay_factor}, lr_decay_at_the_latest {self.args.TTT_lr_decay_at_the_latest}")
        logging.info(f"Backpropagate separately for batches of motion states of size {self.args.TTT_motState_batchSize_per_backprop}.")
        logging.info(f"Use clamp schedule {self.args.TTT_use_clamp_schedule}.")
        
        # # Count decays
        lr_num_decays = 0

        new_phase = False    

        gen_slice_indices = np.random.default_rng(self.args.seed) 

        ksp_shape = masked_corrupted_kspace3D.shape

        for iteration in range(self.args.num_steps_TTT):

            if iteration in self.args.TTT_list_of_split_steps:
                # # Split shots into several equally sized k-space batches (end of first phase)
                logging.info(f"Apply thresholding and split corresponding states into {self.args.TTT_states_per_split} states.")

                # # Apply thresholding
                if iteration > 0:
                    dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state[:,-1]
                else:
                    dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state_init

                dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state > self.args.TTT_DCloss_th_split)[0]
                logging.info(f"Hard DC thresholding applied with th {self.args.TTT_DCloss_th_split}. Num states to split pred_motion_params: {len(dc_th_states_ind)}, indices {dc_th_states_ind}")

                if len(dc_th_states_ind) == 0:
                    logging.info("No states to split. Continue without splitting, reduce total number of steps to 100, and decay lr at step 80.")
                    self.args.num_steps_TTT = 101
                    self.args.TTT_lr_decay_at_the_latest = [80,1000]
                    lr_num_decays = 0
                    self.args.TTT_lr_max_decays = 1
                    self.args.TTT_optimize_all_states_after = None

                else:
                    if iteration > 0:
                        
                        # Save current results in directories for this phase
                        self.evaluate_after_TTT(masked_corrupted_kspace3D, traj, smaps3D_conj, binary_background_mask, optimizer)
                        new_phase = True                    

                    self.args.Ns = self.args.Ns + len(dc_th_states_ind) * (self.args.TTT_states_per_split-1)
                    logging.info(f"New number of states after splitting: {self.args.Ns}")
                    pred_motion_params_split = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
                    traj_split = ([],[])

                    # Go through each old state
                    i_offset = 0
                    self.Ns_list_after_split = []
                    for i in range(len(traj[0])):
                        if i in dc_th_states_ind:
                            # set pred_motion_params in the split states to the average 
                            # of last and next state that are not split states.
                            if i == 0:
                                # use motion parameters of next state that is not in dc_th_states_ind
                                next_state=i+1
                                while next_state in dc_th_states_ind:
                                    next_state += 1
                                if next_state >= len(traj[0]):
                                    pred_motion_params_split[i:i+self.args.TTT_states_per_split,:] = 0
                                else:
                                    pred_motion_params_split[i:i+self.args.TTT_states_per_split,:] = pred_motion_params.data[next_state]
                                i_offset += self.args.TTT_states_per_split-1

                            elif i == len(traj[0])-1:
                                # use motion parameters of last state that is not in dc_th_states_ind
                                last_state=i-1
                                while last_state in dc_th_states_ind:
                                    last_state -= 1
                                if last_state < 0:
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = 0
                                else:
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = pred_motion_params.data[last_state]

                            else:
                                last_state = i-1
                                while last_state in dc_th_states_ind:
                                    last_state -= 1
                                next_state = i+1
                                while next_state in dc_th_states_ind:
                                    next_state += 1
                                
                                if last_state < 0 and next_state >= len(traj[0]):
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = 0
                                elif last_state < 0:
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = pred_motion_params.data[next_state]
                                elif next_state >= len(traj[0]):
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = pred_motion_params.data[last_state]
                                else:
                                    pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = (pred_motion_params.data[last_state] + pred_motion_params.data[next_state]) / 2                                    

                            self.Ns_list_after_split.extend([i+i_offset+j for j in range(self.args.TTT_states_per_split)])
                            i_offset += self.args.TTT_states_per_split-1

                            traj_split[0].extend(list(chunks(traj[0][i], self.args.TTT_states_per_split)))
                            traj_split[1].extend(list(chunks(traj[1][i], self.args.TTT_states_per_split)))


                        else:
                            pred_motion_params_split[i+i_offset,:] = pred_motion_params.data[i]
                            traj_split[0].append(traj[0][i])
                            traj_split[1].append(traj[1][i])

                    logging.info(f"State indices for which we compute gradients after splitting: {self.Ns_list_after_split} in total {len(self.Ns_list_after_split)}")
                    logging.info(f"TTT_all_states_grad_after_split is {self.args.TTT_all_states_grad_after_split}")
                    traj = traj_split
                    pred_motion_params = pred_motion_params_split
                    pred_motion_params_split = None
                    pred_motion_params.requires_grad = True

                    masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.TTT_results_path, dir_name=f"motion_sampling_traj_phase{self.phase+1}", save_figures = True)).cuda(self.args.gpu)
                    
                    if iteration > 0:
                        self.TTT_meters_per_example = self.init_TTT_meters()

                    optimizer = self.init_optimizer(pred_motion_params, self.args.TTT_lr_after_split)

                    self.phase += 1   

            # # Optimize over all states after a certain number of iterations (end of second phase)
            if self.args.TTT_optimize_all_states_after is not None and iteration == self.args.TTT_optimize_all_states_after:
                logging.info(f"Optimize over all {self.args.Ns} motion states in pred_mot_params states for remaining iterations.")
                self.Ns_list_after_split = list(range(0,self.args.Ns))

                # Save current results in directories for this phase
                self.evaluate_after_TTT(masked_corrupted_kspace3D, traj, smaps3D_conj, binary_background_mask, optimizer)
                new_phase = True   

                self.TTT_meters_per_example = self.init_TTT_meters()
                optimizer = self.init_optimizer(pred_motion_params, self.args.TTT_optimize_all_states_after_lr)

                self.phase += 1   


            # # Set grad_translate, grad_rotate to False for the first TTT_num_rot_only_grad_steps iterations
            if iteration < self.args.TTT_num_rot_only_grad_steps:
                grad_translate = False
                grad_rotate = True
            else:
                grad_translate = True
                grad_rotate = True

            optimizer.zero_grad()

            # # Select axis in [0,1,2] for Unet recon in this iteration            
            if self.args.TTT_all_axes and not iteration % 5 == 0 and not new_phase:
                ax_ind = gen_slice_indices.choice(range(3),size=(1), replace=False, )[0]
            else:
                # At the beginning of each phase and in every 5th iteration conduct axial recon
                ax_ind = 2
            
            # # Gradients are only computed during forward/backward pass for a subset of slices
            if self.args.num_slices_per_grad_step == -1:
                rec_id = [ksp_shape[ax_ind+1]//2-1, ksp_shape[ax_ind+1]//2, ksp_shape[ax_ind+1]//2+1]
            else:
                rec_id = gen_slice_indices.choice(range(ksp_shape[ax_ind+1]),size=(self.args.num_slices_per_grad_step), replace=False)                

            # # List of motion states for which we compute gradients
            if self.phase == 0 or self.args.TTT_all_states_grad_after_split or len(self.Ns_list_after_split) == 0:
                Ns_list = list(range(0,self.args.Ns))
            else:
                Ns_list = self.Ns_list_after_split

            # # Potentially reduce GPU memory consumption by splitting the motion states into batches
            if self.args.TTT_motState_batchSize_per_backprop is not None:
                Ns_list_batches = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
            else:
                Ns_list_batches = [Ns_list]

            # # Init logging of DC losses
            self.track_dc_losses_per_state_norm_per_state = torch.zeros(self.args.Ns,1) if iteration == 0 or new_phase else torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1)
            self.track_dc_losses_per_state_norm_all_states = torch.zeros(self.args.Ns,1) if iteration == 0 or new_phase else torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1)
            
            for states_with_grad in Ns_list_batches: 

                #####
                # Step 1: Correct motion corrupted undersampled k-space based on predicted motion parameters
                # states_with_grad potentially contains a batch of motion states of size TTT_motState_batchSize_per_backprop.
                recon = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj,
                                                weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                grad_translate=grad_translate, grad_rotate=grad_rotate, states_with_grad=states_with_grad,
                                                max_coil_size=self.args.TTT_nufft_max_coil_size, shot_ind_maxksp=shot_ind_maxksp) 
                
                recon = complex_mul(recon, smaps3D_conj).sum(dim=0, keepdim=False)

                #####
                # Step 2: Using UNet to predicht the entire volume based on current
                # motion parameters
                recon = unet_forward_all_axes(self.model,recon, rec_id, ax_ind) 
                recon_img3D = recon.detach()
                
                #####
                # Step 3: compute loss to update network parameters      
                # Apply the forward motion corruption (nufft)
                # and compute loss w.r.t. the corrupted k-space.
                recon_coil = complex_mul(recon.unsqueeze(0), smaps3D) 
                recon_ksp = fft2c_ndim(recon_coil, 3)

                recon = motion_corruption_NUFFT(recon_ksp, recon_coil, pred_motion_params, traj, weight_rot=True, args=self.args,
                                                            grad_translate=grad_translate, grad_rotate=grad_rotate, 
                                                            states_with_grad=states_with_grad, 
                                                            max_coil_size=self.args.TTT_nufft_max_coil_size, shot_ind_maxksp=shot_ind_maxksp) 

                loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
                loss.backward()

            # # Update logging of DC losses
            self.update_dc_losses_per_state(recon.detach(), masked_corrupted_kspace3D, masks2D_all_states)

            # # Clamp motion parameters
            if self.args.TTT_use_clamp_schedule:
                if iteration < 15:
                    clamp_val = 5.0
                elif iteration < 30:
                    clamp_val = 8.0
                elif iteration < 45:
                    clamp_val = 10.0
                elif iteration < 60:
                    clamp_val = 12.0
                else:
                    clamp_val = 15.0
            else:
                clamp_val = 15.0
            
            pred_motion_params.data.clamp_(-clamp_val, clamp_val)

            self.evaluate_after_TTT_step(loss.item(), pred_motion_params.detach(), iteration,
                                         recon_img3D.detach(), binary_background_mask, optimizer, 
                                         ax_ind, rec_id, new_phase, traj)
            

            ### call the optimization step
            optimizer.step()
            
            # Custom learning rate decay
            if iteration>self.args.TTT_lr_decay_at_the_latest[lr_num_decays] and lr_num_decays < self.args.TTT_lr_max_decays:
                optimizer.param_groups[0]['lr'] *= self.args.TTT_lr_decay_factor
                lr_num_decays += 1

            new_phase = False

        
        self.evaluate_after_TTT(masked_corrupted_kspace3D, traj, smaps3D_conj, binary_background_mask, optimizer)
        

    def init_optimizer(self, pred_motion_params, lr):
        '''
        Run this after motion parameters are initialized to define the optimizer.
        '''
        
        if self.args.TTT_optimizer == "Adam":
            optimizer = torch.optim.Adam([pred_motion_params], lr=lr)
            logging.info(f"Init Adam optimizer with lr {lr}")
        elif self.args.TTT_optimizer == "SGD":
            optimizer = torch.optim.SGD([pred_motion_params], lr=lr)
            logging.info(f"Init SGD optimizer with lr {lr}")
        else:
            raise ValueError("Unknown optimizer")
        
        return optimizer
    
    def update_dc_losses_per_state(self, recon_kspace3D, masked_corrupted_kspace3D, masks2D_all_states):
        '''
        This function computes the data consistency loss for each state, which can be 
        used for DC loss thresholding after TTT.
        '''
        for i in range(self.args.Ns):
            mask3D_tmp = masks2D_all_states[i]
            dc_loss_norm_per_state = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
            dc_loss_norm_all_states = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            self.track_dc_losses_per_state_norm_per_state[i,-1] = dc_loss_norm_per_state
            self.track_dc_losses_per_state_norm_all_states[i,-1] = dc_loss_norm_all_states
        
    def load_data_init_motion(self):
        '''
        This function must be implemented in the derived class.
        It should load data from a single volume (e.g. from CC359 or invivo).
        '''
        pass

    def init_TTT_meters(self):
        '''
        This function must be implemented in the derived class.
        It should initialize TTT_meters_per_example a dictionary containing 
        TrackMeters for each quantity that should be tracked during TTT.
        '''
        pass

    def evaluate_before_TTT(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance before TTT and must be called by
        load_data_init_motion().
        '''
        pass

    def evaluate_after_TTT_step(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after each TTT step.
        '''
        pass

    def evaluate_after_TTT(self):
        '''
        This function must be implemented in the derived class.
        It should evaluate the performance after TTT.
        '''
        pass



