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

from functions.helpers.helpers_init import init_optimization

from functions.training.losses import SSIMLoss
from functions.helpers.meters import AverageMeter, TrackMeter, TrackMeter_testing
from functions.helpers.progress_bar import ProgressBar
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj, normalize_separate_over_ch

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import unet_forward,motion_correction_NUFFT, generate_random_motion_params, unet_forward_all_axes
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT, gen_rand_mot_params_eventModel


from functions.helpers.helpers_img_metrics import PSNR_torch



def init_TTT_meters():

    TTT_meters_per_example = {
        "TTT_loss" : TrackMeter('decaying'),    
        "L2_motion_parameters" : TrackMeter('decaying'),
        "PSNR_recon_ref" : TrackMeter('increasing'),
        "PSNR_masked_corrected_ref" : TrackMeter('increasing'),
        "PSNR_masked_corrected_masked_ref" : TrackMeter('increasing'),
    } 
            
    return TTT_meters_per_example


class UnetTTTModule():

    def __init__(
            self,
            args,
            model,
            tb_writer,
            ) -> None:
        
        self.args = args
        self.model = model
        self.tb_writer = tb_writer

        self.TTT_meters_per_example = init_TTT_meters()

        self.final_result_dict = {}

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)

    def TTT(self):

        self.model.eval()
        # set requires_grad to False for all model parameters
        for param in self.model.parameters():
            #print(param.requires_grad)
            param.requires_grad = False
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            meter.reset()

        ###############   
        # Load k-space, sensitivity maps and mask
        filepath = os.path.join(self.args.data_drive, self.args.TTT_example_path)
        filename = filepath.split("/")[-1]
        volume_name = filename.split(".")[0]

        smap_file = os.path.join(self.args.data_drive, self.args.TTT_sensmaps_path, "smaps_"+filename)
        with h5py.File(smap_file, 'r') as hf:
            smaps3D = hf['smaps'][()]
        smaps3D = torch.from_numpy(smaps3D)
        smaps3D_conj = complex_conj(smaps3D)
        binary_background_mask = torch.round(torch.sum(complex_mul(smaps3D_conj,smaps3D),0)[:,:,:,0:1])
        binary_background_mask = binary_background_mask.unsqueeze(0)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)
    
        with h5py.File(filepath, "r") as hf:
            ref_kspace3D = hf["kspace"][()]    
        ref_kspace3D = torch.from_numpy(ref_kspace3D)    

        with open(os.path.join(self.args.data_drive, self.args.TTT_mask_path),'rb') as fn:
            mask3D = pickle.load(fn)
            mask3D = torch.tensor(mask3D).unsqueeze(0).unsqueeze(-1) 
            logging.info(f"Using mask from {self.args.TTT_mask_path}")

        # Compute fully sampled and undersampled image volumes and load to gpu
        ref_img3D_coil = ifft2c_ndim(ref_kspace3D, 3)
        ref_img3D = complex_mul(ref_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        masked_kspace3D = ref_kspace3D * mask3D
        

        # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
        # i.e. without batch dimension.
        # Batch dimensions are determined directly before passing through the network
        # and removed directly after the network output.
        masked_kspace3D = masked_kspace3D.cuda(self.args.gpu)
        ref_img3D = ref_img3D.cuda(self.args.gpu)
        ref_img3D_coil = ref_img3D_coil.cuda(self.args.gpu)
        smaps3D = smaps3D.cuda(self.args.gpu)
        smaps3D_conj = smaps3D_conj.cuda(self.args.gpu)
        mask3D = mask3D.cuda(self.args.gpu)
        ref_kspace3D = ref_kspace3D.cuda(self.args.gpu)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)

        ###############
        # Generate sampling trajectory
        traj, masks2D_all_states = generate_interleaved_cartesian_trajectory(self.args.Ns, mask3D, self.args, self.args.TTT_results_path)

        if self.args.TTT_use_nufft_with_dcomp:
            masked_img3D_coil = motion_correction_NUFFT(masked_kspace3D, torch.zeros(self.args.Ns-1, 6).cuda(self.args.gpu), traj, weight_rot=True, args=self.args,
                                                        do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3)
        else:
            masked_img3D_coil = ifft2c_ndim(masked_kspace3D, 3)
        masked_img3D = complex_mul(masked_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        
        ###############
        # Generate Motion State
        self.gt_motion_params = gen_rand_mot_params_eventModel(self.args.Ns-1, self.args.max_trans, self.args.max_rot, self.args.random_motion_seed, self.args.num_motion_events).cuda(self.args.gpu)

        ###############
        # Motion artifact simulation:
        masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, ref_img3D_coil, self.gt_motion_params, traj, weight_rot=True, args=self.args,
                                                            max_coil_size=self.args.TTT_nufft_max_coil_size)
        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        

        ###############
        # Correct motion corrupted undersampled k-space with gt motion parameters
        masked_corrected_img3D_coil = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* self.gt_motion_params, traj, 
                                                              weight_rot=True, args=self.args, do_dcomp=self.args.TTT_use_nufft_with_dcomp, num_iters_dcomp=3,
                                                              max_coil_size=self.args.TTT_nufft_max_coil_size)
        masked_corrected_img3D = complex_mul(masked_corrected_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)
        #masked_corrected_kspace3D = fft2c_ndim(masked_corrected_img3D_coil, 3)

        ###############
        # Log images and metrics before TTT
        list_of_slices = [(0, ref_img3D.shape[0]//2), (1, ref_img3D.shape[1]//2), (2, ref_img3D.shape[2]//2)]
        self.evaluate_before_TTT(ref_img3D, masked_img3D, masked_corrupted_img3D, masked_corrected_img3D, binary_background_mask, 
                                 volume_name, save_slices=True, save_3D = False, list_of_slices = list_of_slices)
        
        
        
        #self.model.train()
        masked_corrupted_img3D = None
        ref_kspace3D = None
        masked_corrupted_img3D_coil = None
        masked_corrected_img3D = None
        #binary_background_mask = None
        masked_corrected_img3D_coil = None
        masked_corrected_img3D = None
        #masked_img3D = None
        #mask3D = None
        masked_kspace3D = None
        masked_img3D_coil = None
        ref_img3D_coil = None

        ###############
        # Init TTT for motion prediction
        pred_motion_params = torch.zeros(self.args.Ns-1, 6).cuda(self.args.gpu)
        #self.pred_motion_params_over_epochs = torch.zeros(self.args.Ns-1, 6, 1)
        pred_motion_params.requires_grad = True
        if self.args.TTT_optimizer == "Adam":
            optimizer = torch.optim.Adam([pred_motion_params], lr=self.args.lr_TTT)
        elif self.args.TTT_optimizer == "SGD":
            optimizer = torch.optim.SGD([pred_motion_params], lr=self.args.lr_TTT)
        else:
            raise ValueError("Unknown optimizer")
        
        with torch.no_grad():
            L2_motion_parameters = torch.sum((pred_motion_params - self.gt_motion_params)**2)
            logging.info(f"Initial L2 motion parameters: {L2_motion_parameters.item()}")
        
        lr_num_decays = 0
        lr_max_decays = 1
        lr_decay_counter = 0
        lr_decay_after = 50
        lr_decay_factor = 0.25
        loss_th = 0.63
        do_another_lr_decay = True
        grad_translate = True
        grad_rotate = True

        logging.info(f"Motion parameters are max rot {self.args.max_rot}, max trans {self.args.max_trans}, motion seed {self.args.random_motion_seed}, motion states {self.args.Ns} and motion events {self.args.num_motion_events}.")
        logging.info(f"Number of slices per grad step: {self.args.num_slices_per_grad_step}")

        logging.info(f"number of rot only grad steps: {self.args.TTT_num_rot_only_grad_steps}")
        logging.info(f"lr_num_decays {lr_num_decays}, lr_max_decays {lr_max_decays}, lr_decay_counter {lr_decay_counter}, lr_decay_after {lr_decay_after}, lr_decay_factor {lr_decay_factor}, do_another_lr_decay {do_another_lr_decay}")
        logging.info(f"Optimize only over motion parameters in motion corruption steps {self.args.TTT_only_motCorrupt_grad}.")
        logging.info(f"Backpropagate separately for batches of motion states of size {self.args.TTT_motState_batchSize_per_backprop}.")

        logging.info(f"Use clamp schedule {self.args.TTT_use_clamp_schedule}.")

        
        for iteration in range(self.args.num_steps_TTT):

            if iteration < self.args.TTT_num_rot_only_grad_steps:
                grad_translate = False
                grad_rotate = True
            else:
                grad_translate = True
                grad_rotate = True

            # !!!
            # Implement TTT for motion correction
            optimizer.zero_grad()

            # chose random ax_ind from [0,1,2]
            if self.args.TTT_all_axes:
                ax_ind = np.random.choice(range(3),size=(1), replace=False)[0]
            else:
                ax_ind = 2
            
            if self.args.num_slices_per_grad_step == -1:
                rec_id = [masked_corrupted_kspace3D.shape[ax_ind+1]//2-1, masked_corrupted_kspace3D.shape[ax_ind+1]//2, masked_corrupted_kspace3D.shape[ax_ind+1]//2+1]
            else:
                rec_id = np.random.choice(range(masked_corrupted_kspace3D.shape[ax_ind+1]),size=(self.args.num_slices_per_grad_step), replace=False)  # The picked slice                
        

            #self.print_gpu_memory_usage(step_index=0)
            

            Ns_list = list(range(0,self.args.Ns-1))

            if self.args.TTT_only_motCorrupt_grad is False:
                if self.args.TTT_motState_batchSize_per_backprop is not None:
                    Ns_list_batches_motCorrect = [Ns_list[i:i+self.args.TTT_motState_batchSize_per_backprop] for i in range(0, len(Ns_list), self.args.TTT_motState_batchSize_per_backprop)]
                else:
                    Ns_list_batches_motCorrect = [Ns_list]
            else:
                Ns_list_batches_motCorrect = [Ns_list]

            losses = []
            self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            
            for states_with_grad_motCorrect in Ns_list_batches_motCorrect: # for loop if TTT_only_motCorrupt_grad is False else only one iteration

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
                masked_corrected_img3D = recon.detach()

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

                for j,states_with_grad_motCorrupt in enumerate(Ns_list_batches_motCorrupt):
                    
                    # select traj_tmp and motion parameters to apply motion corruption to
                    if self.args.TTT_only_motCorrupt_grad is False:
                        # Always include zero state in addition to the states with grad
                        traj_tmp = ([traj[0][0]], [traj[1][0]])
                        zero_state_in_recon = True
                        for i in states_with_grad_motCorrupt:
                            traj_tmp[0].append(traj[0][i+1])
                            traj_tmp[1].append(traj[1][i+1])
                    else:
                        # Do not include zero state
                        zero_state_in_recon = False
                        traj_tmp = ([traj[0][i+1] for i in states_with_grad_motCorrupt], [traj[1][i+1] for i in states_with_grad_motCorrupt])
                        
                    pred_motion_params_tmp = pred_motion_params[states_with_grad_motCorrupt, :]


                    recon = motion_corruption_NUFFT(recon_ksp, recon_coil, pred_motion_params_tmp, traj_tmp, weight_rot=True, args=self.args,
                                                                grad_translate=grad_translate, grad_rotate=grad_rotate,max_coil_size=self.args.TTT_nufft_max_coil_size)
                    mask3D_tmp = torch.zeros_like(mask3D)
                    # set mask3D_tmp to 1 where the recon is 1
                    mask3D_tmp[complex_abs(recon[0:1,:,:,:]).unsqueeze(-1)>0] = 1.0
                    loss = torch.sum(torch.abs(recon-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
                    loss.backward()
                    losses.append(loss.item())

                    self.update_dc_losses_per_state(recon.detach(), masked_corrupted_kspace3D, masks2D_all_states, states_with_grad_motCorrupt, zero_state_in_recon)

            loss = torch.tensor(losses).mean()

            #self.print_gpu_memory_usage(step_index=3)

            if iteration==0:
                init_loss = loss.item()

            if iteration == 0:
                if init_loss > loss_th:
                    logging.info(f"Initial loss is high (larger {loss_th}). Setting lr to 4.0 and allow 2 lr decays.")
                    optimizer.param_groups[0]['lr'] = 4.0
                    lr_max_decays = 2
                else:
                    logging.info(f"Initial loss is low (smaller {loss_th}). Setting lr to 1.0 and allow 1 lr decays.")
                    optimizer.param_groups[0]['lr'] = 1.0
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

            self.evaluate_after_TTT_step(loss.item(), pred_motion_params.detach(), iteration, losses,
                                         recon_img3D.detach(), ref_img3D, masked_img3D, masked_corrected_img3D.detach(), binary_background_mask, optimizer,
                                         ax_ind, rec_id)
            

            ### call the optimization step
            optimizer.step()

            if init_loss > loss.item() and do_another_lr_decay:
                lr_decay_counter += 1
                if lr_decay_counter == lr_decay_after:
                    optimizer.param_groups[0]['lr'] *= lr_decay_factor
                    lr_num_decays += 1
                    lr_decay_counter = 0
                    if lr_num_decays == lr_max_decays: 
                        do_another_lr_decay = False

        
        self.evaluate_after_TTT(masked_corrupted_kspace3D, traj, smaps3D_conj, binary_background_mask, volume_name)

    def update_dc_losses_per_state(self, recon_kspace3D, masked_corrupted_kspace3D, masks2D_all_states, states_with_grad_motCorrupt, zero_state_in_recon):
        if zero_state_in_recon:
            mask3D_tmp = masks2D_all_states[0]
            dc_loss_norm_per_state = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
            dc_loss_norm_all_states = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            self.track_dc_losses_per_state_norm_per_state[0,-1] = dc_loss_norm_per_state
            self.track_dc_losses_per_state_norm_all_states[0,-1] = dc_loss_norm_all_states
        for i in states_with_grad_motCorrupt:
            mask3D_tmp = masks2D_all_states[i+1]
            dc_loss_norm_per_state = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D*mask3D_tmp))
            dc_loss_norm_all_states = torch.sum(torch.abs(recon_kspace3D*mask3D_tmp-masked_corrupted_kspace3D*mask3D_tmp)) / torch.sum(torch.abs(masked_corrupted_kspace3D))
            self.track_dc_losses_per_state_norm_per_state[i+1,-1] = dc_loss_norm_per_state
            self.track_dc_losses_per_state_norm_all_states[i+1,-1] = dc_loss_norm_all_states


    def evaluate_after_TTT_step(self,TTT_loss, pred_motion_params, iteration, losses, recon_img3D, ref_img3D, masked_img3D, masked_corrected_img3D, binary_background_mask, optimizer, ax_ind, rec_ind):
        # !!!
        
        self.TTT_meters_per_example["TTT_loss"].update(TTT_loss, iteration)

        

        L2_motion_parameters = torch.sum((pred_motion_params - self.gt_motion_params)**2)
        self.TTT_meters_per_example["L2_motion_parameters"].update(L2_motion_parameters, iteration)

        self.pred_motion_params_over_epochs = torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1) if iteration > 0 else pred_motion_params.cpu().unsqueeze(-1)

        if recon_img3D is not None:
            recon_img3D = complex_abs(recon_img3D * binary_background_mask)
            ref_img3D = complex_abs(ref_img3D * binary_background_mask)
            masked_img3D = complex_abs(masked_img3D * binary_background_mask)
            masked_corrected_img3D = complex_abs(masked_corrected_img3D * binary_background_mask)

            self.TTT_meters_per_example["PSNR_recon_ref"].update(PSNR_torch(recon_img3D, ref_img3D), iteration)
            self.TTT_meters_per_example["PSNR_masked_corrected_ref"].update(PSNR_torch(masked_corrected_img3D, ref_img3D), iteration)
            self.TTT_meters_per_example["PSNR_masked_corrected_masked_ref"].update(PSNR_torch(masked_corrected_img3D, masked_img3D), iteration)

            ### Save reconstruction at minimum dc loss
            if TTT_loss==self.TTT_meters_per_example["TTT_loss"].best_val:
                recon_img3D = recon_img3D.cpu()
                torch.save(recon_img3D, self.args.TTT_results_path+"/reconstruction_min_reconDC_loss.pt")
                list_of_slices = None
                save_slice_images_from_volume(recon_img3D[0], list_of_slices, self.args.TTT_results_path, "last_step_recon_TTT", axis_names = ["coronal","saggital","axial"])

        else:
            self.TTT_meters_per_example["PSNR_recon_ref"].update(0, iteration)
            self.TTT_meters_per_example["PSNR_masked_corrected_ref"].update(0, iteration)
            self.TTT_meters_per_example["PSNR_masked_corrected_masked_ref"].update(0, iteration)

        pickle.dump( self.TTT_meters_per_example, open( os.path.join(self.args.TTT_results_path_numerical, 'TTT_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        np.save(os.path.join(self.args.TTT_results_path_numerical, 'pred_motion_params_over_epochs.npy'), self.pred_motion_params_over_epochs.cpu().numpy())

        text_to_log = f"Epoch {iteration} | "
        for name,meter in zip(self.TTT_meters_per_example.keys(), self.TTT_meters_per_example.values()):
            text_to_log += f"{name}: {meter.val[-1]:.5f} | "

        text_to_log += f"lr: {optimizer.param_groups[0]['lr']:.5e} | AxInd {ax_ind} | SliceInd {rec_ind}"
        logging.info(text_to_log)


        N_s = self.pred_motion_params_over_epochs.shape[0]+1
        pred_motion_params = pred_motion_params.cpu().numpy()

        scale_down_factor = 0.02

        self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params, N_s, scale_down_factor, dc_loss_ind = -1, fig_name=None)
        if iteration % 20 == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params, N_s, scale_down_factor, dc_loss_ind = -1, fig_name=f"motion_pred_params_{iteration}.png")


    def evaluate_after_TTT(self, masked_corrupted_kspace3D=None, traj=None, smaps3D_conj=None, binary_background_mask=None, volume_name=None):
        N_s = self.pred_motion_params_over_epochs.shape[0]+1
        best_step = self.TTT_meters_per_example["TTT_loss"].best_epoch

        # The DC loss in the 0th iter corresponds to the all zeto motion state
        # Hence, no best_step+1 is needed
        pred_motion_params_final = self.pred_motion_params_over_epochs[:,:,best_step].numpy() 

        self.final_result_dict["TTT_loss"] = self.TTT_meters_per_example["TTT_loss"].best_val
        self.final_result_dict["TTT_best_step"] = best_step 
        self.final_result_dict["L2_motion_parameters"] = self.TTT_meters_per_example["L2_motion_parameters"].val[best_step]
        self.final_result_dict["PSNR_ref_vs_recon_TTT"] = self.TTT_meters_per_example["PSNR_recon_ref"].val[best_step]
        self.final_result_dict["pred_motion_params_final"] = pred_motion_params_final
        self.final_result_dict["gt_motion_params"] = self.gt_motion_params.cpu().numpy()
        pickle.dump( self.final_result_dict, open( os.path.join(self.args.TTT_results_path_numerical, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.TTT_results_path_numerical, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.TTT_results_path_numerical, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        if masked_corrupted_kspace3D is not None:
            with torch.no_grad():
                list_of_slices = None
                input = motion_correction_NUFFT(masked_corrupted_kspace3D, -1* torch.from_numpy(pred_motion_params_final), traj, weight_rot=True, args=self.args,
                                                max_coil_size=self.args.TTT_nufft_max_coil_size) # masked_corrected_img3D_coil
                masked_corrected_img3D = complex_mul(input, smaps3D_conj).sum(dim=0, keepdim=False) # masked_corrected_img3D
                recon_img3D_axial = self.reconstruct_volume_slicewise(masked_corrected_img3D.unsqueeze(0), recon_axis=3)
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_TTT", axis_names = ["coronal","saggital","axial"])

        scale_down_factor = 0.02
        self.save_figure_motion_parameters_dc_loss(best_step, pred_motion_params_final, N_s, scale_down_factor, dc_loss_ind = best_step)

        logging.info(f"Best step (min consistency loss): {best_step}")
        logging.info(f"Gt motion parameters: {self.gt_motion_params}")
        logging.info(f"Motion prediction at best step: {pred_motion_params_final}")
        logging.info(f"L2 motion parameters at best step: {self.TTT_meters_per_example['L2_motion_parameters'].val[best_step]}")
        logging.info(f"PSNR reference vs. recon at best step: {self.TTT_meters_per_example['PSNR_recon_ref'].val[best_step]}")
        logging.info(f"PSNR reference vs. masked corrected at best step: {self.TTT_meters_per_example['PSNR_masked_corrected_ref'].val[best_step]}")
        logging.info(f"PSNR masked corrected vs. masked at best step: {self.TTT_meters_per_example['PSNR_masked_corrected_masked_ref'].val[best_step]}")


        logging.info(f"Best L2 motion parameters: {self.TTT_meters_per_example['L2_motion_parameters'].best_val} at step {self.TTT_meters_per_example['L2_motion_parameters'].best_epoch}")
        logging.info(f"Best PSNR reference vs. recon: {self.TTT_meters_per_example['PSNR_recon_ref'].best_val} at step {self.TTT_meters_per_example['PSNR_recon_ref'].best_epoch}")
        logging.info(f"Best PSNR reference vs. masked corrected: {self.TTT_meters_per_example['PSNR_masked_corrected_ref'].best_val} at step {self.TTT_meters_per_example['PSNR_masked_corrected_ref'].best_epoch}")
        logging.info(f"Best PSNR masked corrected vs. masked: {self.TTT_meters_per_example['PSNR_masked_corrected_masked_ref'].best_val} at step {self.TTT_meters_per_example['PSNR_masked_corrected_masked_ref'].best_epoch}")

        

    def reconstruct_volume_batchwise(self, masked_img3D, recon_axis):
        '''
        Reconstruc 3D volume from slice wise with a 2D network, where all slices
        are processed in one batch.
            - masked_img3D: 4D tensor (X, Y, Z, 2)
            - recon_axis: int,  0,1,2
        '''
        assert recon_axis in [0,1,2], "Recon axis must be 0,1 or 2"
        assert len(masked_img3D.shape) == 4, "Input must be 4D tensor (X, Y, Z, 2)"

        masked_img3D = torch.moveaxis(masked_img3D, recon_axis, 0)

        masked_img3D_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(masked_img3D,-1,1), eps=1e-11)
        recon_img3D = self.model(masked_img3D_norm)
        recon_img3D = recon_img3D * std + mean
        recon_img3D = torch.moveaxis(recon_img3D, 1, -1)   

        recon_img3D = torch.moveaxis(recon_img3D, 0, recon_axis)

        return recon_img3D
    
    
    def reconstruct_slices_into_volume(self, masked_img3D, recon_img3D, axes_slice_pairs):
        '''
        Reconstruct 2D slices specified in axes_slice_pairs and insert them into 
        the 3D volume recon_img3D.
            - masked_img3D: 4D tensor (X, Y, Z, 2)
            - recon_img3D: 4D tensor (X, Y, Z, 2)
            - axes_slice_pairs: list of tuples [(axis, slice), (axis, slice), ...]
        '''
        assert len(masked_img3D.shape) == 4, "Input must be 4D tensor (X, Y, Z, 2)"
        assert len(recon_img3D.shape) == 4, "Input must be 4D tensor (X, Y, Z, 2)"

        for axis, slice in axes_slice_pairs:
            assert axis in [0,1,2], "Axis must be 0,1 or 2"
            assert slice < masked_img3D.shape[axis], "Slice index out of bounds"

            masked_img3D = torch.moveaxis(masked_img3D, axis, 0)
            recon_img3D = torch.moveaxis(recon_img3D, axis, 0)

            masked_img2D = masked_img3D[slice].moveaxis(-1,0).unsqueeze(0)
            masked_img2D_norm, mean, std = normalize_separate_over_ch(masked_img2D, eps=1e-11)  
            recon_img2D = self.model(masked_img2D_norm)
            recon_img2D = recon_img2D * std + mean
            recon_img2D = torch.moveaxis(recon_img2D, 1, -1)

            recon_img3D[slice] = recon_img2D[0]

            masked_img3D = torch.moveaxis(masked_img3D, 0, axis)
            recon_img3D = torch.moveaxis(recon_img3D, 0, axis)

        return recon_img3D
    

    def evaluate_before_TTT(self, ref_img3D, masked_img3D, masked_corrupted_img3D, masked_corrected_img3D, binary_background_mask, volume_name, save_slices=False, save_3D = False, list_of_slices = None):
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
            ######
            # Inspect fully sampled reference volume vs. undersampled volume

            if save_slices:
                save_slice_images_from_volume(ref_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "ref", axis_names = ["coronal","saggital","axial"])
                save_slice_images_from_volume(masked_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked", axis_names = ["coronal","saggital","axial"])
                save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrupted", axis_names = ["coronal","saggital","axial"])
                save_slice_images_from_volume(masked_corrected_img3D.cpu(), list_of_slices, self.args.TTT_results_path, "masked_corrected", axis_names = ["coronal","saggital","axial"])

            if save_3D:
                volume_to_k3d_html(complex_abs(ref_img3D).cpu().numpy(), volume_name = "ref", save_path = self.args.TTT_results_path)
                volume_to_k3d_html(complex_abs(masked_img3D).cpu().numpy(), volume_name = "masked", save_path = self.args.TTT_results_path)
                volume_to_k3d_html(complex_abs(masked_corrupted_img3D).cpu().numpy(), volume_name = "masked_corrupted", save_path = self.args.TTT_results_path)
                volume_to_k3d_html(complex_abs(masked_corrected_img3D).cpu().numpy(), volume_name = "masked_corrected", save_path = self.args.TTT_results_path)

            psnr = PSNR_torch(complex_abs(masked_img3D), complex_abs(masked_corrected_img3D))
            logging.info(f"PSNR undersampled vs. undersampled corrected: {psnr}")
            self.final_result_dict["PSNR_zf_motfree_vs_zf_corrected_gtmot"] = psnr

            psnr = PSNR_torch(complex_abs(masked_img3D), complex_abs(ref_img3D))
            logging.info(f"PSNR reference vs. undersampled: {psnr}")
            self.final_result_dict["PSNR_reference_vs_zf_motfree"] = psnr

            psnr = PSNR_torch(complex_abs(masked_corrupted_img3D), complex_abs(ref_img3D))
            logging.info(f"PSNR reference vs. undersampled corrupted: {psnr}")
            self.final_result_dict["PSNR_reference_vs_zf_corrupted"] = psnr

            psnr = PSNR_torch(complex_abs(masked_corrected_img3D), complex_abs(ref_img3D))
            logging.info(f"PSNR reference vs. undersampled corrected: {psnr}")
            self.final_result_dict["PSNR_reference_vs_zf_corrected"] = psnr


            ######
            # Reconstruct undersampled volume with 2D network along different axes

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = self.reconstruct_volume_slicewise(masked_img3D.unsqueeze(0), recon_axis=3)
                #logging.info(f"PSNR reference vs. recon axial: {PSNR_torch(complex_abs(recon_img3D_axial), complex_abs(ref_img3D))}")
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon axial (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_motfree_axial_binmasked"] = psnr

            if recon_saggital:
                # saggital reconstruction
                recon_img3D_saggital = self.reconstruct_volume_slicewise(masked_img3D.unsqueeze(0), recon_axis=2)
                #logging.info(f"PSNR reference vs. recon saggital: {PSNR_torch(complex_abs(recon_img3D_saggital), complex_abs(ref_img3D))}")
                recon_img3D_saggital_fg = recon_img3D_saggital * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon saggital (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_motfree_saggital_binmasked"] = psnr

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = self.reconstruct_volume_slicewise(masked_img3D.unsqueeze(0), recon_axis=1)
                #logging.info(f"PSNR reference vs. recon coronal: {PSNR_torch(complex_abs(recon_img3D_coronal), complex_abs(ref_img3D))}")
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon coronal (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_motfree_coronal_binmasked"] = psnr

            # PSNR between axial/saggital/coronal reconstructions
            #logging.info(f"PSNR axial vs. saggital (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_saggital_fg))}")
            #logging.info(f"PSNR axial vs. coronal (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_coronal_fg))}")
            #logging.info(f"PSNR saggital vs. coronal (masked): {PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(recon_img3D_coronal_fg))}")

            if save_slices:
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_masked", axis_names = ["coronal","saggital","axial"]) if recon_axial else None
                save_slice_images_from_volume(recon_img3D_saggital_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_saggital_masked", axis_names = ["coronal","saggital","axial"]) if recon_saggital else None
                save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_masked", axis_names = ["coronal","saggital","axial"]) if recon_coronal else None

            if save_3D:
                volume_to_k3d_html(complex_abs(recon_img3D_axial_fg[0]).cpu().numpy(), volume_name = "recon_axial_masked", save_path = self.args.TTT_results_path) if recon_axial else None
                volume_to_k3d_html(complex_abs(recon_img3D_saggital_fg[0]).cpu().numpy(), volume_name = "recon_saggital_masked", save_path = self.args.TTT_results_path) if recon_saggital else None
                volume_to_k3d_html(complex_abs(recon_img3D_coronal_fg[0]).cpu().numpy(), volume_name = "recon_coronal_masked", save_path = self.args.TTT_results_path) if recon_coronal else None

            ########
            # Reconstruct undersampled corrupted volume with 2D network along different axes

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = self.reconstruct_volume_slicewise(masked_corrupted_img3D.unsqueeze(0), recon_axis=3)
                #logging.info(f"PSNR reference vs. recon axial corrupted: {PSNR_torch(complex_abs(recon_img3D_axial), complex_abs(ref_img3D))}")
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon axial corrupted (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_axial_binmasked"] = psnr

            # saggital reconstruction
            if recon_saggital:
                recon_img3D_saggital = self.reconstruct_volume_slicewise(masked_corrupted_img3D.unsqueeze(0), recon_axis=2)
                #logging.info(f"PSNR reference vs. recon saggital corrupted: {PSNR_torch(complex_abs(recon_img3D_saggital), complex_abs(ref_img3D))}")
                recon_img3D_saggital_fg = recon_img3D_saggital * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon saggital corrupted (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_saggital_binmasked"] = psnr

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = self.reconstruct_volume_slicewise(masked_corrupted_img3D.unsqueeze(0), recon_axis=1)
                #logging.info(f"PSNR reference vs. recon coronal corrupted: {PSNR_torch(complex_abs(recon_img3D_coronal), complex_abs(ref_img3D))}")
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon coronal corrupted (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrupted_coronal_binmasked"] = psnr

            # PSNR between axial/saggital/coronal reconstructions
            #logging.info(f"PSNR axial vs. saggital corrupted (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_saggital_fg))}")
            #logging.info(f"PSNR axial vs. coronal corrupted (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_coronal_fg))}")
            #logging.info(f"PSNR saggital vs. coronal corrupted (masked): {PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(recon_img3D_coronal_fg))}")

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

            if recon_axial:
                # axial reconstruction
                recon_img3D_axial = self.reconstruct_volume_slicewise(masked_corrected_img3D.unsqueeze(0), recon_axis=3)
                #logging.info(f"PSNR reference vs. recon axial corrected: {PSNR_torch(complex_abs(recon_img3D_axial), complex_abs(ref_img3D))}")
                recon_img3D_axial_fg = recon_img3D_axial * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon axial corrected (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_axial_binmasked"] = psnr

            if recon_saggital:
                # saggital reconstruction
                recon_img3D_saggital = self.reconstruct_volume_slicewise(masked_corrected_img3D.unsqueeze(0), recon_axis=2)
                #logging.info(f"PSNR reference vs. recon saggital corrected: {PSNR_torch(complex_abs(recon_img3D_saggital), complex_abs(ref_img3D))}")
                recon_img3D_saggital_fg = recon_img3D_saggital * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon saggital corrected (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_saggital_binmasked"] = psnr

            if recon_coronal:  
                # coronal reconstruction
                recon_img3D_coronal = self.reconstruct_volume_slicewise(masked_corrected_img3D.unsqueeze(0), recon_axis=1)
                #logging.info(f"PSNR reference vs. recon coronal corrected: {PSNR_torch(complex_abs(recon_img3D_coronal), complex_abs(ref_img3D))}")
                recon_img3D_coronal_fg = recon_img3D_coronal * binary_background_mask

                psnr = PSNR_torch(complex_abs(recon_img3D_coronal_fg), complex_abs(ref_img3D))
                logging.info(f"PSNR reference vs. recon coronal corrected (masked): {psnr}")
                self.final_result_dict["PSNR_reference_vs_recon_corrected_coronal_binmasked"] = psnr

            # PSNR between axial/saggital/coronal reconstructions
            #logging.info(f"PSNR axial vs. saggital corrected (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_saggital_fg))}")
            #logging.info(f"PSNR axial vs. coronal corrected (masked): {PSNR_torch(complex_abs(recon_img3D_axial_fg), complex_abs(recon_img3D_coronal_fg))}")
            #logging.info(f"PSNR saggital vs. coronal corrected (masked): {PSNR_torch(complex_abs(recon_img3D_saggital_fg), complex_abs(recon_img3D_coronal_fg))}")

            if save_slices:
                save_slice_images_from_volume(recon_img3D_axial_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_axial_corrected_masked", axis_names = ["coronal","saggital","axial"]) if recon_axial else None
                save_slice_images_from_volume(recon_img3D_saggital_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_saggital_corrected_masked", axis_names = ["coronal","saggital","axial"]) if recon_saggital else None
                save_slice_images_from_volume(recon_img3D_coronal_fg[0].cpu(), list_of_slices, self.args.TTT_results_path, "recon_coronal_corrected_masked", axis_names = ["coronal","saggital","axial"]) if recon_coronal else None

            if save_3D:
                volume_to_k3d_html(complex_abs(recon_img3D_axial_fg[0]).cpu().numpy(), volume_name = "recon_axial_corrected_masked", save_path = self.args.TTT_results_path) if recon_axial else None
                volume_to_k3d_html(complex_abs(recon_img3D_saggital_fg[0]).cpu().numpy(), volume_name = "recon_saggital_corrected_masked", save_path = self.args.TTT_results_path) if recon_saggital else None
                volume_to_k3d_html(complex_abs(recon_img3D_coronal_fg[0]).cpu().numpy(), volume_name = "recon_coronal_corrected_masked", save_path = self.args.TTT_results_path) if recon_coronal else None

            pickle.dump( self.final_result_dict, open( os.path.join(self.args.TTT_results_path_numerical, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        
    
    def reconstruct_volume_slicewise(self, masked_img3D, recon_axis):
        '''
        Reconstruc 3D volume from slice wise with a 2D network.
            - masked_img3D: 5D tensor (1, X, Y, Z, 2)
            - recon_axis: int,  1, 2, 3
        '''
        assert recon_axis in [1,2,3], "Recon axis must be 1, 2 or 3"
        assert len(masked_img3D.shape) == 5, "Input must be 5D tensor (1, X, Y, Z, 2)"

        recon_img3D = masked_img3D.clone()
        for slice in range(masked_img3D.shape[recon_axis]):
            if recon_axis == 1:
                img_slice = masked_img3D[:,slice,:,:,:]
            elif recon_axis == 2:
                img_slice = masked_img3D[:,:,slice,:,:]
            elif recon_axis == 3:
                img_slice = masked_img3D[:,:,:,slice,:]

            img_slice_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(img_slice,-1,1), eps=1e-11)
            img_slice_recon = self.model(img_slice_norm)
            img_slice_recon = img_slice_recon * std + mean
            img_slice_recon = torch.moveaxis(img_slice_recon, 1, -1)   

            if recon_axis == 1:
                recon_img3D[:,slice,:,:,:] = img_slice_recon
            elif recon_axis == 2:
                recon_img3D[:,:,slice,:,:] = img_slice_recon
            elif recon_axis == 3:
                recon_img3D[:,:,:,slice,:] = img_slice_recon

        return recon_img3D

    def print_gpu_memory_usage(self, step_index):

        current_memory = torch.cuda.memory_allocated(self.args.gpu)  # device_id is the ID of your GPU
        peak_memory = torch.cuda.max_memory_allocated(self.args.gpu)
        current_memory_reserved = torch.cuda.memory_reserved(self.args.gpu)
        peak_memory_reserved = torch.cuda.max_memory_reserved(self.args.gpu)
        logging.info(f"{step_index} Current GPU memory usage: {current_memory / 1024**3} GB")
        logging.info(f"{step_index} Peak GPU memory usage: {peak_memory / 1024**3} GB")
        logging.info(f"{step_index} Current GPU memory reserved: {current_memory_reserved / 1024**3} GB")
        logging.info(f"{step_index} Peak GPU memory reserved: {peak_memory_reserved / 1024**3} GB")


    def save_figure_motion_parameters_dc_loss(self, iteration, pred_motion_params, N_s, scale_down_factor=0.02, dc_loss_ind=-1, fig_name=None):

        save_dir = os.path.join(self.args.TTT_results_path, "motion_param_figures")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.plot(range(1,N_s,1),pred_motion_params[:,0])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,0].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"x axis translation iter {iteration}")

        plt.subplot(232)
        plt.plot(range(1,N_s,1),pred_motion_params[:,1])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,1].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"y axis translation iter {iteration}")

        plt.subplot(233)
        plt.plot(range(1,N_s,1),pred_motion_params[:,2])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,2].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"z axis translation iter {iteration}")

        plt.subplot(234)
        plt.plot(range(1,N_s,1),pred_motion_params[:,3])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,3].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"x-y plane roatation iter {iteration}")

        plt.subplot(235)
        plt.plot(range(1,N_s,1),pred_motion_params[:,4])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,4].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"y-z plane roatation iter {iteration}")

        plt.subplot(236)
        plt.plot(range(1,N_s,1),pred_motion_params[:,5])
        plt.plot(range(1,N_s,1),self.gt_motion_params[:,5].cpu().numpy())
        plt.ylabel('mot params in deg or mm')
        plt.xlabel('Motion states over time')
        plt.legend(['Prediction', 'Ground truth'], loc='lower left')
        plt.twinx()
        plt.plot(range(0,N_s,1),scale_down_factor*self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
        plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_all_states[:,dc_loss_ind].cpu().numpy(), 'g')
        plt.ylabel('DC loss of recon k-space')
        plt.legend(['DC norm per state', 'DC norm all states'], loc='lower right')
        plt.title(f"x-z plane roatation iter {iteration}")
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()


    
            
