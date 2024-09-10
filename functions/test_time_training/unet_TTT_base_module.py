import torch
import logging
import numpy as np
import os
import pickle
import scipy.signal as signal

from functions.motion_simulation.motion_functions import motion_correction_NUFFT, motion_corruption_NUFFT, unet_forward_all_axes
from functions.motion_simulation.motion_functions import DC_loss_thresholding, get_maxKsp_shot
from functions.helpers.helpers_math import complex_mul, fft2c_ndim, complex_abs, chunks

from functions.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories, print_gpu_memory_usage, save_figure_original_resolution

from functions.helpers.helpers_init import init_logging, initialize_directories_TTT

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

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
            #print(param.requires_grad)
            param.requires_grad = False
            
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            meter.reset()

        # # Load data
        traj, smaps3D, smaps3D_conj, binary_background_mask, masked_corrupted_kspace3D, mask3D = self.load_data_init_motion(evaluate_before_TTT=True)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.TTT_results_path, dir_name=f"motion_sampling_traj_phase{self.phase}", save_figures = True)).cuda(self.args.gpu)

        # # Compute a normalization mask that divides the k-space of each shot by the energy of that shot
        if self.args.TTT_norm_per_shot:
            mask_norm_per_shot = torch.zeros((1, mask3D.shape[1], mask3D.shape[2], 1, 1)).cuda(self.args.gpu)

            for i in range(self.args.Ns):
                energy_per_shot = torch.sum(torch.abs(masked_corrupted_kspace3D * masks2D_all_states[i]))
                mask_norm_per_shot += masks2D_all_states[i] /energy_per_shot
            
            save_path = os.path.join(self.args.TTT_results_path, f"stats_per_state_phase{self.phase}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # create and save figure of mask_norm_per_shot
            save_figure_original_resolution(mask_norm_per_shot[0,:,:,0,0].cpu().numpy(), save_path, f"mask_norm_per_shot_Ns{self.args.Ns}")

        ###############
        # Init TTT for motion prediction
        # We assume no motion takes place during the first shot for both motionTraj_simMot equal
        # 'uniform_interShot_event_model' and 'uniform_intraShot_event_model'
        # Hence, to use fix_mot_maxksp_shot we only need to assure that the center is in the first shot
            
        if self.args.TTT_sampTraj_simMot == 'linear_cartesian' and self.args.fix_mot_maxksp_shot:
            raise ValueError("fix_mot_maxksp_shot is not supported for linear_cartesian sampling trajectory as center is not in the first shot.")
        if not self.args.center_in_first_state and self.args.fix_mot_maxksp_shot:
            raise ValueError("fix_mot_maxksp_shot is not supported for center_in_first_state=False.")
        
        if self.loaded_pred_motion_params is None:
            logging.info("Initialize pred_motion_params with all zeros.")
            pred_motion_params = torch.zeros(self.args.Ns, 6).cuda(self.args.gpu)
        else:
            logging.info("Initialize pred_motion_params with loaded_pred_motion_params.")
            pred_motion_params = self.loaded_pred_motion_params.clone()

        shot_ind_maxksp = get_maxKsp_shot(masked_corrupted_kspace3D, traj, self.args.fix_mot_maxksp_shot)

        if self.args.fix_mot_maxksp_shot and self.gt_motion_params is not None:
            #if shot_ind_maxksp != 0 and self.args.motionTraj_simMot != "uniform_interShot_event_model":
            #    raise ValueError("fix_mot_maxksp_shot with shot_ind_maxksp not equal 0 is only supported for uniform_interShot_event_model.")
            
            # Set pred_motion_params for shot with max k-space energy to gt_motion_params
            pred_motion_params[shot_ind_maxksp] = self.gt_motion_params[shot_ind_maxksp]
            logging.info(f"Set pred_motion_params for shot with max k-space energy to gt_motion_params: {pred_motion_params[shot_ind_maxksp].cpu().numpy()}")

        pred_motion_params.requires_grad = True

        optimizer = self.init_optimizer(pred_motion_params, self.args.lr_TTT)

        logging.info(f"fix_mot_maxksp_shot={self.args.fix_mot_maxksp_shot} and shot index with max k-space energy is {shot_ind_maxksp}.")
        logging.info(f"number of rot only grad steps: {self.args.TTT_num_rot_only_grad_steps}")
        logging.info(f"lr_max_decays {self.args.TTT_lr_max_decays}, lr_decay_after {self.args.TTT_lr_decay_after}, lr_decay_factor {self.args.TTT_lr_decay_factor}, lr_decay_at_the_latest {self.args.TTT_lr_decay_at_the_latest}")
        logging.info(f"Backpropagate separately for batches of motion states of size {self.args.TTT_motState_batchSize_per_backprop}.")
        logging.info(f"Use clamp schedule {self.args.TTT_use_clamp_schedule}.")
        
        # # Count decays
        lr_num_decays = 0

        new_phase = False    

        gen_slice_indices = np.random.default_rng(self.args.seed) 

        ksp_shape = masked_corrupted_kspace3D.shape

        for iteration in range(self.args.num_steps_TTT):

            # # Reset motion parameters with large dc loss
            if iteration in self.args.TTT_list_of_reset_steps:
                # # Apply peak or hard thresholding
                dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state[:,-1]
                if self.args.TTT_set_DCloss_lr_th == "peak":
                    dc_th_states_ind = signal.find_peaks(dc_loss_per_state_norm_per_state)[0]
                    logging.info(f"Peak DC thresholding applied. Num states to set pred_motion_params to zero: {len(dc_th_states_ind)}")
                else:
                    dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state > self.args.TTT_set_DCloss_lr_th)[0]
                    logging.info(f"Hard DC thresholding applied. Num states to set pred_motion_params to zero: {len(dc_th_states_ind)}")

                pred_motion_params.data[dc_th_states_ind] = 0
                optimizer = self.init_optimizer(pred_motion_params, 1)


            # # Split shots into several equally sized k-space batches. To this end
            if iteration in self.args.TTT_list_of_split_steps:
                logging.info(f"Apply thresholding and split corresponding states into {self.args.TTT_states_per_split} states.")

                # # Apply peak or hard thresholding
                if iteration > 0:
                    dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state[:,-1]
                else:
                    dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state_init

                if self.args.TTT_set_DCloss_lr_th == "peak":
                    dc_th_states_ind = signal.find_peaks(dc_loss_per_state_norm_per_state)[0]
                    logging.info(f"Peak DC thresholding applied. Num states to split pred_motion_params: {len(dc_th_states_ind)}")
                else:
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
                            # set pred_motion_params in the split states to the average of last and next state that are not split states
                            if i == 0:
                                # use motion parameters of next state that is not in dc_th_states_ind
                                j=i+1
                                while j in dc_th_states_ind:
                                    j += 1
                                pred_motion_params_split[i:i+self.args.TTT_states_per_split,:] = pred_motion_params.data[j]
                            elif i == len(traj[0])-1:
                                # use motion parameters of last state that is not in dc_th_states_ind
                                j=i-1
                                while j in dc_th_states_ind:
                                    j -= 1
                                pred_motion_params_split[i:i+self.args.TTT_states_per_split,:] = pred_motion_params.data[j]
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
                                    
                            
                            
                            #pred_motion_params_split[i+i_offset:i+i_offset+self.args.TTT_states_per_split,:] = 0

                            self.Ns_list_after_split.extend([i+i_offset+j for j in range(self.args.TTT_states_per_split)])
                            i_offset += self.args.TTT_states_per_split-1

                            # nl = int(np.ceil(len(traj[0][i]) / self.args.TTT_states_per_split))
                            # traj_split[0].extend([traj[0][i][j*nl:(j+1)*nl] for j in range(self.args.TTT_states_per_split)])
                            # traj_split[1].extend([traj[1][i][j*nl:(j+1)*nl] for j in range(self.args.TTT_states_per_split)])

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
                        #self.pred_motion_params_over_epochs = pred_motion_params.detach().cpu().unsqueeze(-1)
                        self.TTT_meters_per_example = self.init_TTT_meters()

                    optimizer = self.init_optimizer(pred_motion_params, self.args.TTT_lr_after_split)
                    # load optimizer from save_path
                    #optimizer.load_state_dict(torch.load(os.path.join(save_path, 'optimizer.pth')))

                    self.phase += 1   

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
                ax_ind = 2
            
            if self.args.num_slices_per_grad_step == -1:
                rec_id = [ksp_shape[ax_ind+1]//2-1, ksp_shape[ax_ind+1]//2, ksp_shape[ax_ind+1]//2+1]
            else:
                rec_id = gen_slice_indices.choice(range(ksp_shape[ax_ind+1]),size=(self.args.num_slices_per_grad_step), replace=False)                
        
            #self.print_gpu_memory_usage(step_index=0)

            if self.phase == 0 or self.args.TTT_all_states_grad_after_split or len(self.Ns_list_after_split) == 0:
                Ns_list = list(range(0,self.args.Ns))
            else:
                Ns_list = self.Ns_list_after_split

            if self.args.TTT_only_motCorrupt_grad is False:
                if self.args.TTT_motState_batchSize_per_backprop is not None:
                    Ns_list_batches_motCorrect = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
                else:
                    Ns_list_batches_motCorrect = [Ns_list]
            else:
                Ns_list_batches_motCorrect = [Ns_list]

            #self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            self.track_dc_losses_per_state_norm_per_state = torch.zeros(self.args.Ns,1) if iteration == 0 or new_phase else torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1)
            #self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            self.track_dc_losses_per_state_norm_all_states = torch.zeros(self.args.Ns,1) if iteration == 0 or new_phase else torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1)
            
            for states_with_grad_motCorrect in Ns_list_batches_motCorrect: # for loop if TTT_only_motCorrupt_grad is False else only one iteration

                #####
                # Step 1: Correct motion corrupted undersampled k-space based on predicted motion parameters
                if self.args.TTT_only_motCorrupt_grad:
                    # states_with_grad contains all states but since we set grad_translate, grad_rotate to False
                    # we do not compute gradients here in the motion correctoin step
                    recon = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj,
                                                    weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                    grad_translate=False, grad_rotate=False, states_with_grad=states_with_grad_motCorrect,
                                                    max_coil_size=self.args.TTT_nufft_max_coil_size, shot_ind_maxksp=shot_ind_maxksp) # masked_corrected_img3D_coil
                else:
                    # states_with_grad potentially contains a batch of motion states of size TTT_motState_batchSize_per_backprop.
                    # Here we compute gradients also for the motion correction step according to the current value of grad_translate, grad_rotate
                    recon = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj,
                                                    weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                    grad_translate=grad_translate, grad_rotate=grad_rotate, states_with_grad=states_with_grad_motCorrect,
                                                    max_coil_size=self.args.TTT_nufft_max_coil_size, shot_ind_maxksp=shot_ind_maxksp) # masked_corrected_img3D_coil
                                    
                recon = complex_mul(recon, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D

                #self.print_gpu_memory_usage(step_index=1)

                #####
                # Step 2: Using UNet to predicht the entire volume based on current
                # motion parameters
                recon = unet_forward_all_axes(self.model,recon, rec_id, ax_ind) # recon_img3D
                recon_img3D = recon.detach()
                
                #####
                # Step 3: compute loss to update network parameters      
                # Option 1: Apply the forward motion corruption (nufft)
                # and compute loss w.r.t. the corrupted k-space.
                recon_coil = complex_mul(recon.unsqueeze(0), smaps3D) # recon_img3D_coil
                recon_ksp = fft2c_ndim(recon_coil, 3) # recon_kspace3D

                #self.print_gpu_memory_usage(step_index=2)

                if self.args.TTT_only_motCorrupt_grad is False:
                    Ns_list_batches_motCorrupt = [states_with_grad_motCorrect]
                else:
                    Ns_list_batches_motCorrupt = []
                    if self.args.TTT_motState_batchSize_per_backprop is not None:
                        Ns_list_batches_motCorrupt = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
                    else:
                        Ns_list_batches_motCorrupt = [Ns_list]

                for states_with_grad_motCorrupt in Ns_list_batches_motCorrupt:

                    recon = motion_corruption_NUFFT(recon_ksp, recon_coil, pred_motion_params, traj, weight_rot=True, args=self.args,
                                                                grad_translate=grad_translate, grad_rotate=grad_rotate, 
                                                                states_with_grad=states_with_grad_motCorrupt, 
                                                                max_coil_size=self.args.TTT_nufft_max_coil_size, shot_ind_maxksp=shot_ind_maxksp) 
                    if self.args.TTT_norm_per_shot:
                        loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D)*mask_norm_per_shot) / self.args.Ns
                    else:
                        loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
                    loss.backward()

            self.update_dc_losses_per_state(recon.detach(), masked_corrupted_kspace3D, masks2D_all_states, states_with_grad_motCorrupt)

            #self.print_gpu_memory_usage(step_index=3)

            if iteration==0:
                init_loss = loss.item()

            if iteration == 0 and self.args.TTT_set_DCloss_lr:
                if init_loss > self.args.TTT_set_DCloss_lr_th:
                    self.args.TTT_lr_max_decays += 1
                    #logging.info(f"Initial loss is high (larger {self.args.TTT_set_DCloss_lr_th}). Setting lr to 4.0 and allow an additional lr decay.")
                    logging.info(f"Initial loss {init_loss:.5f} is high (larger {self.args.TTT_set_DCloss_lr_th}). Setting lr to 4.0 and allow {self.args.TTT_lr_max_decays} lr decays in phase 0.")
                    optimizer.param_groups[0]['lr'] = 4.0
                    self.args.TTT_lr_decay_at_the_latest = [50, 90, 120, 200]
                else:
                    logging.info(f"Initial loss is low (smaller {self.args.TTT_set_DCloss_lr_th}). Setting lr to 1.0 and allow {self.args.TTT_lr_max_decays} lr decays in phase 0.")
                    optimizer.param_groups[0]['lr'] = 1.0
                    self.args.TTT_lr_decay_at_the_latest = [80,120, 200]
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

            if self.args.fix_mot_maxksp_shot and self.gt_motion_params is not None:
                pred_motion_params.data[shot_ind_maxksp] = self.gt_motion_params[shot_ind_maxksp]

            self.evaluate_after_TTT_step(loss.item(), pred_motion_params.detach(), iteration,
                                         recon_img3D.detach(), binary_background_mask, optimizer, 
                                         ax_ind, rec_id, new_phase, traj)
            

            ### call the optimization step
            optimizer.step()
            #self.track_grad_per_state = torch.cat((self.track_grad_per_state, pred_motion_params.grad.cpu().unsqueeze(-1)), dim=-1) if iteration > 0 else pred_motion_params.grad.cpu().unsqueeze(-1)
            self.track_grad_per_state = pred_motion_params.grad.cpu().unsqueeze(-1) if iteration == 0 or new_phase else torch.cat((self.track_grad_per_state, pred_motion_params.grad.cpu().unsqueeze(-1)), dim=-1)
            
            # if (init_loss > loss.item() or iteration>self.args.TTT_lr_decay_at_the_latest) and lr_num_decays < self.args.TTT_lr_max_decays:
            #     lr_decay_counter += 1
            #     if lr_decay_counter == self.args.TTT_lr_decay_after or iteration>self.args.TTT_lr_decay_at_the_latest:
            if iteration>self.args.TTT_lr_decay_at_the_latest[lr_num_decays] and lr_num_decays < self.args.TTT_lr_max_decays:
                optimizer.param_groups[0]['lr'] *= self.args.TTT_lr_decay_factor
                lr_num_decays += 1
                #lr_decay_counter = 0
                #self.args.TTT_lr_decay_at_the_latest = 90 #self.args.TTT_lr_decay_at_the_latest*10

            new_phase = False

        
        self.evaluate_after_TTT(masked_corrupted_kspace3D, traj, smaps3D_conj, binary_background_mask, optimizer)
        

    def finetune_after_DCTh(self):
        '''
        This function can be called after TTT. It performs DC loss thresholding to remove states with 
        incorrectly estimated motion parameters. Then the predicted motion parameters of the remaining states
        are fine-tuned.
        '''

        if self.args.fix_mot_maxksp_shot:
            raise ValueError("fix_mot_maxksp_shot is not supported for finetune_after_DCTh.")
        
        # # Load final_result_dict to access previous pred_motion_params, traj, dc loss per state and gt_motion_param
        motion_TTT_results_path = os.path.join(self.args.TTT_results_path_numerical, 'final_result_dict.pkl')
        with open(motion_TTT_results_path,'rb') as fn:
            #final_results_dict_TTT = pickle.load(fn)
            final_results_dict_TTT = CPU_Unpickler(fn).load()

        # # Update args.TTT_results_path and args.TTT_log_path
        self.args = initialize_directories_TTT(self.args, TTT_fintune=True)
        init_logging(self.args.TTT_log_path)

        logging.info(f"Start finetuning after DC loss thresholding with exp name {self.args.experiment_name_TTT_finetune}.")
        logging.info(f"Load pred_motion_params from motion TTT from {motion_TTT_results_path}")

        # # Load data
        _, smaps3D, smaps3D_conj, binary_background_mask, masked_corrupted_kspace3D, mask3D = self.load_data_init_motion(evaluate_before_TTT=False)

        pred_motion_params = torch.from_numpy(final_results_dict_TTT['pred_motion_params_final']).cuda(self.args.gpu)
        traj = final_results_dict_TTT['traj']
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.TTT_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu)

        dc_loss_per_state_norm_per_state = final_results_dict_TTT['dc_losses_per_state_norm_per_state_min_reconDC_loss']
        if final_results_dict_TTT['gt_motion_params'] is not None:
            gt_motion_params = torch.from_numpy(final_results_dict_TTT['gt_motion_params']).cuda(self.args.gpu)
        else:
            gt_motion_params = None

        # # Perform thresholding based on dc loss and update traj, gt_motion_params (if available), pred_motion_params
        masked_corrupted_kspace3D, traj, gt_motion_params, pred_motion_params, masks2D_all_states, Ns, _, _ = DC_loss_thresholding(dc_loss_per_state_norm_per_state, self.args.altopt_dc_threshold, 
                                                                                                                             traj, gt_motion_params, pred_motion_params, 
                                                                                                                             masks2D_all_states, masked_corrupted_kspace3D)
        if Ns == self.args.Ns:
            logging.info("Abort fine-tuning since no states removed after DC loss thresholding.")
            return
        self.args.Ns = Ns
        self.gt_motion_params = gt_motion_params

        # # Logging before TTT
        with torch.no_grad():
            # Inspect corrupted image after thresholding
            masked_corrupted_img3D_coil = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* torch.zeros(self.args.Ns, 6).cuda(self.args.gpu), traj, 
                                                                weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                                max_coil_size=self.args.TTT_nufft_max_coil_size)
            masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

            # Inspect corrected image after thresholding
            masked_corrected_img3D_coil = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj, 
                                                              weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                              max_coil_size=self.args.TTT_nufft_max_coil_size)
            masked_corrected_img3D = complex_mul(masked_corrected_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

            self.evaluate_before_TTT(masked_corrupted_img3D, binary_background_mask, 
                                    masked_corrected_img3D=masked_corrected_img3D, 
                                    smaps3D=smaps3D, traj=traj, smaps3D_conj=smaps3D_conj,
                                    masked_corrupted_kspace3D=masked_corrupted_kspace3D,
                                    save_slices=True, save_3D = False, list_of_slices = None)
            
            masked_corrupted_img3D_coil = None
            masked_corrected_img3D_coil = None
            masked_corrected_img3D = None
            masked_corrupted_img3D = None
        
        # # Run TTT again with the updated pred_motion_params
        self.model.eval()
        # set requires_grad to False for all model parameters
        for param in self.model.parameters():
            #print(param.requires_grad)
            param.requires_grad = False
        pred_motion_params.requires_grad = True
        optimizer = self.init_optimizer(pred_motion_params, self.args.lr_TTT_finetune)
        self.TTT_meters_per_example = self.init_TTT_meters()
        self.final_result_dict = {}

        ksp_shape = masked_corrupted_kspace3D.shape

        for iteration in range(self.args.num_steps_TTT_finetune):
            grad_translate = True
            grad_rotate = True

            optimizer.zero_grad()

            # # Select axis in [0,1,2] for Unet recon in this iteration            
            if self.args.TTT_all_axes and not iteration % 5 == 0:
                ax_ind = np.random.choice(range(3),size=(1), replace=False)[0]
            else:
                ax_ind = 2
            
            if self.args.num_slices_per_grad_step == -1:
                rec_id = [ksp_shape[ax_ind+1]//2-1, ksp_shape[ax_ind+1]//2, ksp_shape[ax_ind+1]//2+1]
            else:
                rec_id = np.random.choice(range(ksp_shape[ax_ind+1]),size=(self.args.num_slices_per_grad_step), replace=False)                  
        
            #self.print_gpu_memory_usage(step_index=0)

            Ns_list = list(range(0,self.args.Ns))

            if self.args.TTT_only_motCorrupt_grad is False:
                if self.args.TTT_motState_batchSize_per_backprop is not None:
                    Ns_list_batches_motCorrect = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
                else:
                    Ns_list_batches_motCorrect = [Ns_list]
            else:
                Ns_list_batches_motCorrect = [Ns_list]

            self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            for states_with_grad_motCorrect in Ns_list_batches_motCorrect: # for loop if TTT_only_motCorrupt_grad is False else only one iteration

                #####
                # Step 1: Correct motion corrupted undersampled k-space based on predicted motion parameters
                if self.args.TTT_only_motCorrupt_grad:
                    # states_with_grad contains all states but since we set grad_translate, grad_rotate to False
                    # we do not compute gradients here in the motion correctoin step
                    recon = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj,
                                                    weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                    grad_translate=False, grad_rotate=False, states_with_grad=states_with_grad_motCorrect,
                                                    max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                else:
                    # states_with_grad potentially contains a batch of motion states of size TTT_motState_batchSize_per_backprop.
                    # Here we compute gradients also for the motion correction step according to the current value of grad_translate, grad_rotate
                    recon = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* pred_motion_params, traj,
                                                    weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                    grad_translate=grad_translate, grad_rotate=grad_rotate, states_with_grad=states_with_grad_motCorrect,
                                                    max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                                    
                recon = complex_mul(recon, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D

                #self.print_gpu_memory_usage(step_index=1)

                #####
                # Step 2: Using UNet to predicht the entire volume based on current
                # motion parameters
                recon = unet_forward_all_axes(self.model,recon, rec_id, ax_ind) # recon_img3D
                recon_img3D = recon.detach()
                
                #####
                # Step 3: compute loss to update network parameters      
                # Option 1: Apply the forward motion corruption (nufft)
                # and compute loss w.r.t. the corrupted k-space.
                recon_coil = complex_mul(recon.unsqueeze(0), smaps3D) # recon_img3D_coil
                recon_ksp = fft2c_ndim(recon_coil, 3) # recon_kspace3D

                #self.print_gpu_memory_usage(step_index=2)

                if self.args.TTT_only_motCorrupt_grad is False:
                    Ns_list_batches_motCorrupt = [states_with_grad_motCorrect]
                else:
                    Ns_list_batches_motCorrupt = []
                    if self.args.TTT_motState_batchSize_per_backprop is not None:
                        Ns_list_batches_motCorrupt = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
                    else:
                        Ns_list_batches_motCorrupt = [Ns_list]

                for states_with_grad_motCorrupt in Ns_list_batches_motCorrupt:

                    recon = motion_corruption_NUFFT(recon_ksp, recon_coil, pred_motion_params, traj, weight_rot=True, args=self.args,
                                                                grad_translate=grad_translate, grad_rotate=grad_rotate, 
                                                                states_with_grad=states_with_grad_motCorrupt, max_coil_size=self.args.TTT_nufft_max_coil_size) 
                    loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
                    loss.backward()

                    self.update_dc_losses_per_state(recon.detach(), masked_corrupted_kspace3D, masks2D_all_states, states_with_grad_motCorrupt)

            #self.print_gpu_memory_usage(step_index=3)

            clamp_val = 15.0
            pred_motion_params.data.clamp_(-clamp_val, clamp_val)

            self.evaluate_after_TTT_step(loss.item(), pred_motion_params.detach(), iteration,
                                         recon_img3D.detach(), binary_background_mask, optimizer, 
                                         ax_ind, rec_id)
            

            ### call the optimization step
            optimizer.step()
        
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
    
    def update_dc_losses_per_state(self, recon_kspace3D, masked_corrupted_kspace3D, masks2D_all_states, states_with_grad_motCorrupt):
        '''
        This function computes the data consistency loss for each stat, which can be 
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
        The following quantities HAVE to be returned to run TTT_step (on gpu):
            - traj: Sampling trajectory
            - smaps3D: Sensitivity maps, torch.Tensor (coils,x,y,z,2)
            - smaps3D_conj: Complex conjugate of smaps3D
            - binary_background_mask: Binary mask torch.Tensor (x,y,z,1)
            - masked_corruped_kspace3D: Motion-corrupted k-space torch.Tensor (coils,x,y,z,2)
            - mask3D: Mask torch.Tensor (1,x,y,z,1)
        Optional quantities:
            - gt_motion_params: Ground truth motion parameters, torch.Tensor (Ns,6)
            - ref_img3D: Reference image, torch.Tensor (x,y,z,2)
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



