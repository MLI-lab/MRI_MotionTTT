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
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj

from functions.helpers.helpers_log_save_image_utils import save_figure, save_figure_original_resolution, save_masks_from_motion_sampling_trajectories
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html

from functions.motion_simulation.motion_functions import unet_forward,motion_correction_NUFFT, generate_random_motion_params, motion_alignment
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT, gen_rand_mot_params_eventModel


from functions.helpers.helpers_img_metrics import PSNR_torch
from torch.autograd import Variable
import ptwt, pywt


def init_alt_opt_meters():

    alt_opt_meters_per_example = {
        "recon_loss_total" : TrackMeter('decaying'),    
        "recon_loss_dc" : TrackMeter('decaying'),
        "recon_loss_reg" : TrackMeter('decaying'),
        "motion_loss" : TrackMeter('decaying'),
        "L2_gt_motion_parameters" : TrackMeter('decaying'),
        "PSNR_recon_ref" : TrackMeter('increasing'),
    } 
            
    return alt_opt_meters_per_example

class AltOptModule():

    def __init__(
            self,
            args,
            ) -> None:
        
        self.args = args

        self.alt_opt_meters_per_example = init_alt_opt_meters()

        self.ssim_loss = SSIMLoss(gpu=self.args.gpu)


    def alt_opt(self):
        for name,meter in zip(self.alt_opt_meters_per_example.keys(), self.alt_opt_meters_per_example.values()):
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
        masked_img3D_coil = ifft2c_ndim(masked_kspace3D, 3)
        masked_img3D = complex_mul(masked_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
        # i.e. without batch dimension.
        # Batch dimensions are determined directly before passing through the network
        # and removed directly after the network output.
        masked_img3D = masked_img3D.cuda(self.args.gpu)
        ref_img3D = ref_img3D.cuda(self.args.gpu)
        ref_img3D_coil = ref_img3D_coil.cuda(self.args.gpu)
        smaps3D = smaps3D.cuda(self.args.gpu)
        smaps3D_conj = smaps3D_conj.cuda(self.args.gpu)
        mask3D = mask3D.cuda(self.args.gpu)
        ref_kspace3D = ref_kspace3D.cuda(self.args.gpu)
        binary_background_mask = binary_background_mask.cuda(self.args.gpu)

        ###############
        # Generate sampling trajectory
        traj, _ = generate_interleaved_cartesian_trajectory(self.args.Ns, mask3D, self.args)

        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask3D[0,:,:,0,0].cpu().numpy() , save_path=self.args.altopt_results_path, dir_name="motion_sampling_traj", save_figures = False)).cuda(self.args.gpu)

                
        ###############
        # Generate Motion State
        self.gt_motion_params = gen_rand_mot_params_eventModel(self.args.Ns-1, self.args.max_trans, self.args.max_rot, self.args.random_motion_seed, self.args.num_motion_events).cuda(self.args.gpu)

        ###############
        # Motion artifact simulation:

        masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, ref_img3D_coil, self.gt_motion_params, traj, 
                                                            weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size)

        self.log_before_alt_opt(masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask)
        
        

        ###############
        # Init Reconstruction Volume
        mse = torch.nn.MSELoss()
        recon = Variable(torch.zeros(ref_img3D.shape)).cuda(self.args.gpu)
        recon.data.uniform_(0,1)

        if self.args.altopt_motion_estimation_only:
            recon = ref_img3D.clone()
        
        if self.args.altopt_optimizer_recon == "Adam":
            optimizer_recon = torch.optim.Adam([recon],lr=self.args.altopt_lr_recon)
        elif self.args.altopt_optimizer_recon == "SGD":
            optimizer_recon = torch.optim.SGD([recon],lr=self.args.altopt_lr_recon)
        

        ###############
        # Init motion parameter estimation
        pred_motion_params = torch.zeros(self.args.Ns-1, 6).cuda(self.args.gpu)
        if self.args.altopt_recon_only and self.args.altopt_recon_only_with_motionKnowledge:
            logging.info("Using ground truth motion parameters for reconstruction.")
            pred_motion_params = self.gt_motion_params

        if self.args.alt_opt_on_TTTexp:
            motion_TTT_results_path = os.path.join(self.args.TTT_results_path_numerical, 'final_result_dict.pkl')
            logging.info(f"Load results from motion TTT from {motion_TTT_results_path}")
            with open(motion_TTT_results_path,'rb') as fn:
                final_results_dict_TTT = pickle.load(fn)
            pred_motion_params = torch.from_numpy(final_results_dict_TTT['pred_motion_params_final']).cuda(self.args.gpu)
        elif self.args.alt_opt_on_alt_opt_exp:
            altopt_results_path = os.path.join(self.args.altopt_load_path, 'motion_params_min_reconDC_loss.pt')
            logging.info(f"Load motion parameters from {altopt_results_path}")
            pred_motion_params = torch.load(altopt_results_path).cuda(self.args.gpu)

        self.pred_motion_params_over_epochs = torch.zeros(self.args.Ns-1, 6, 1)

        if self.args.altopt_dc_thresholding and self.args.altopt_recon_only and (self.args.alt_opt_on_TTTexp or self.args.alt_opt_on_alt_opt_exp):
            if self.args.alt_opt_on_TTTexp:
                motion_TTT_results_path_dc_losses = os.path.join(self.args.TTT_results_path_numerical, 'track_dc_losses_per_state_norm_per_state.pkl')
                logging.info(f"Load dc losses from motion TTT from {motion_TTT_results_path_dc_losses}")
                with open(motion_TTT_results_path_dc_losses,'rb') as fn:
                    track_dc_losses_per_state_norm_per_state = pickle.load(fn)
                TTT_best_step = final_results_dict_TTT['TTT_best_step']
                dc_loss_per_state_norm_per_state = track_dc_losses_per_state_norm_per_state[:,TTT_best_step]
            elif self.args.alt_opt_on_alt_opt_exp:
                altopt_results_path = os.path.join(self.args.altopt_load_path, 'reconstruction_min_reconDC_loss.pt')
                logging.info(f"Load reconstruction from {altopt_results_path}")
                recon_old = torch.load(altopt_results_path).cuda(self.args.gpu)
                recon_old_coil = complex_mul(recon_old, smaps3D)
                recon_old_kspace3d_coil = fft2c_ndim(recon_old_coil, 3)
                recon_old_kspace3d_coil = motion_corruption_NUFFT(recon_old_kspace3d_coil, recon_old_coil, pred_motion_params, traj, 
                                                                  weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size)
                
                self.track_dc_losses_per_state_norm_per_state = torch.zeros(self.args.Ns,1)
                self.track_dc_losses_per_state_norm_all_states = torch.zeros(self.args.Ns,1)
                Ns_list = list(range(0,self.gt_motion_params.shape[0]))
                self.update_dc_losses_per_state(recon_old_kspace3d_coil.detach(), masked_corrupted_kspace3D, masks2D_all_states, Ns_list, zero_state_in_recon=True)
                dc_loss_per_state_norm_per_state = self.track_dc_losses_per_state_norm_per_state.cpu().numpy()


            dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state[1:] < self.args.altopt_dc_threshold)[0]

            pred_motion_params = pred_motion_params[dc_th_states_ind]
            self.gt_motion_params = self.gt_motion_params[dc_th_states_ind]
            self.args.Ns = self.gt_motion_params.shape[0]+1
            traj= ([traj[0][0]] + [traj[0][i+1] for i in dc_th_states_ind], [traj[1][0]] + [traj[1][i+1] for i in dc_th_states_ind])

            masks2D_all_states = torch.cat((masks2D_all_states[0:1], masks2D_all_states[dc_th_states_ind+1]), dim=0)
            
            masked_corrupted_kspace3D = motion_corruption_NUFFT(ref_kspace3D, ref_img3D_coil, self.gt_motion_params, traj, 
                                                            weight_rot=True, args=self.args, max_coil_size=self.args.altopt_nufft_max_coil_size)
            
        if self.args.altopt_align_motParams and self.args.altopt_recon_only and (self.args.alt_opt_on_TTTexp or self.args.alt_opt_on_alt_opt_exp):
            pred_motion_params = motion_alignment(pred_motion_params, self.gt_motion_params, r=10, num_points=5001, gpu=self.args.gpu) 
                

        if self.args.altopt_optimizer_motion == "Adam":
            optimizer_motion = torch.optim.Adam([pred_motion_params], lr=self.args.altopt_lr_motion)
        elif self.args.altopt_optimizer_motion == "SGD":
            optimizer_motion = torch.optim.SGD([pred_motion_params], lr=self.args.altopt_lr_motion)

        logging.info(f"""Starting Alt Opt with {self.args.altopt_steps_total} total steps, 
                     {self.args.altopt_steps_recon} recon steps with lr {self.args.altopt_lr_recon:.1e}, lambda {self.args.altopt_lam_recon:.1e} and optimizer {self.args.altopt_optimizer_recon},
                     and {self.args.altopt_steps_motion} motion est steps with lr {self.args.altopt_lr_motion:.1e} and optimizer {self.args.altopt_optimizer_motion}.
                     Motion parameter settings are mot events {self.args.num_motion_events} max_rot/max_trans {self.args.max_rot}/{self.args.max_trans}, random motion seed {self.args.random_motion_seed} and Ns {self.args.Ns}.
                     Recon only is {self.args.altopt_recon_only} (with gt motion {self.args.altopt_recon_only_with_motionKnowledge}) and motion est only is {self.args.altopt_motion_estimation_only}.""")
        
        ref_kspace3D = None
        ref_img3D_coil = None

        total_steps = 0
        for iteration in range(self.args.altopt_steps_total):
            
            if not self.args.altopt_motion_estimation_only:
                recon.requires_grad = True
                pred_motion_params.requires_grad = False
                for recon_step in range(self.args.altopt_steps_recon):
                    if self.args.altopt_recon_only:
                        self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
                        self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            
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

                    recon_for_eval = recon.detach()*binary_background_mask
                    self.evaluate_after_recon_step(recon_for_eval, pred_motion_params, ref_img3D, loss_recon.item(), loss_reg.item(), loss_dc.item(), iteration, recon_step, total_steps)

                    if self.args.altopt_recon_only:
                        Ns_list = list(range(0,self.gt_motion_params.shape[0]))
                        self.update_dc_losses_per_state(recon_kspace3d_coil.detach(), masked_corrupted_kspace3D, masks2D_all_states, Ns_list, zero_state_in_recon=True)
                        self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.detach().cpu(), self.args.Ns, scale_down_factor=0.02, dc_loss_ind=-1, fig_name='motion_pred_params.png')
                        if iteration % 30 == 0:
                            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.detach().cpu(), self.args.Ns, scale_down_factor=0.02, dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')
        

            if not self.args.altopt_recon_only:
                self.track_dc_losses_per_state_norm_per_state = torch.cat((self.track_dc_losses_per_state_norm_per_state, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
                self.track_dc_losses_per_state_norm_all_states = torch.cat((self.track_dc_losses_per_state_norm_all_states, torch.zeros(self.args.Ns,1)), dim=-1) if iteration > 0 else torch.zeros(self.args.Ns,1)
            
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

                    loss_motion = mse( recon_kspace3d_coil_corrupted , masked_corrupted_kspace3D )
                    loss_motion.backward()
                    optimizer_motion.step()

                    self.evaluate_after_motion_step(pred_motion_params.detach(), loss_motion.item(), iteration, motion_step, total_steps)

                Ns_list = list(range(0,self.args.Ns-1))
                self.update_dc_losses_per_state(recon_kspace3d_coil_corrupted.detach(), masked_corrupted_kspace3D, masks2D_all_states, Ns_list, zero_state_in_recon=True)
                self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.detach().cpu(), self.args.Ns, scale_down_factor=0.02, dc_loss_ind=-1, fig_name='motion_pred_params.png')
                if iteration % 30 == 0:
                    self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.detach().cpu(), self.args.Ns, scale_down_factor=0.02, dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')
        
        self.evaluate_after_alt_opt(ref_img3D)
                    
    def log_before_alt_opt(self,masked_corrupted_kspace3D, masked_kspace3D, ref_kspace3D, ref_img3D, masked_img3D, smaps3D_conj, binary_background_mask):

        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        list_of_slices = None        
        save_slice_images_from_volume(ref_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "ref_img", axis_names = ["coronal","saggital","axial"])
        save_slice_images_from_volume(masked_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_img", axis_names = ["coronal","saggital","axial"])
        save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_corrupted_img", axis_names = ["coronal","saggital","axial"])

        psnr = PSNR_torch(complex_abs(masked_img3D), complex_abs(ref_img3D))
        logging.info(f"PSNR reference vs. undersampled: {psnr}")

        psnr = PSNR_torch(complex_abs(masked_corrupted_img3D), complex_abs(ref_img3D))
        logging.info(f"PSNR reference vs. undersampled corrupted: {psnr}")


    def evaluate_after_recon_step(self, recon, pred_motion_params, ref_img3D, loss_recon, loss_reg, loss_dc, iteration, recon_step, total_steps):

        self.alt_opt_meters_per_example["recon_loss_total"].update(loss_recon, total_steps)
        self.alt_opt_meters_per_example["recon_loss_dc"].update(loss_dc, total_steps)
        self.alt_opt_meters_per_example["recon_loss_reg"].update(loss_reg, total_steps)
            
        psnr_step = PSNR_torch(complex_abs(recon), complex_abs(ref_img3D))
        self.alt_opt_meters_per_example["PSNR_recon_ref"].update(psnr_step, total_steps)
                
        ### Log the data
        logging.info(f"total step {total_steps}, iteration {iteration}, recon step {recon_step} -- recon loss: {loss_recon:.5f} | DC loss: {loss_dc:.5f} | Reg loss: {loss_reg:.5f} | PSNR: {psnr_step:.5f}")

        ### Save reconstruction at minimum dc loss
        if loss_dc==self.alt_opt_meters_per_example["recon_loss_dc"].best_val:
            torch.save(recon.cpu(), self.args.altopt_results_path+"/reconstruction_min_reconDC_loss.pt")
            torch.save(pred_motion_params.cpu(), self.args.altopt_results_path+"/motion_params_min_reconDC_loss.pt")

        if psnr_step == self.alt_opt_meters_per_example["PSNR_recon_ref"].best_val:
            torch.save(recon.cpu(), self.args.altopt_results_path+"/reconstruction_best_PSNR.pt")
            torch.save(pred_motion_params.cpu(), self.args.altopt_results_path+"/motion_params_best_PSNR.pt")
            

    def evaluate_after_motion_step(self, pred_motion_params, loss_motion, iteration, motion_step, total_steps):

        self.alt_opt_meters_per_example["motion_loss"].update(loss_motion, total_steps)
        L2_motion_parameters = torch.sum((pred_motion_params - self.gt_motion_params)**2)
        self.alt_opt_meters_per_example["L2_gt_motion_parameters"].update(L2_motion_parameters, total_steps)
        
        self.pred_motion_params_over_epochs = torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1)

        ### Log the data
        logging.info(f"total step {total_steps}, iteration {iteration}, motion step {motion_step} -- motion loss: {loss_motion:.5f} | L2(gt_motion_params): {L2_motion_parameters:.5f}")


    def evaluate_after_alt_opt(self, ref_img3D):
        N_s = self.gt_motion_params.shape[0]+1

        pickle.dump( self.alt_opt_meters_per_example, open( os.path.join(self.args.altopt_results_path, 'alt_opt_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        if not self.args.altopt_motion_estimation_only:
            # load the best reconstruction according to DC loss
            recon = torch.load(self.args.altopt_results_path+"/reconstruction_min_reconDC_loss.pt")
            recon = recon.detach().cuda(self.args.gpu)
            psnr = PSNR_torch(complex_abs(recon), complex_abs(ref_img3D))
            logging.info(f"PSNR of the best reconstruction according to min recon DC loss: {psnr}")

            list_of_slices = None
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.altopt_results_path, "recon_axial_altOpt_min_reconDC_loss", axis_names = ["coronal","saggital","axial"])

            # Load motion parameters from the best reconstruction according to DC loss
            pred_motion_params_final = torch.load(self.args.altopt_results_path+"/motion_params_min_reconDC_loss.pt").numpy()
            L2_motion_parameters_final = np.sum((pred_motion_params_final - self.gt_motion_params.cpu().numpy())**2)
            logging.info(f"L2(gt_motion_params) of the best reconstruction according to min recon DC loss: {L2_motion_parameters_final}")
        else:
            pred_motion_params_final = self.pred_motion_params_over_epochs[:,:,-1].numpy()
            L2_motion_parameters_final = np.sum((pred_motion_params_final - self.gt_motion_params.cpu().numpy())**2)
            logging.info(f"L2(gt_motion_params) at last step of motion estimation only: {L2_motion_parameters_final}")

        plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.plot(range(N_s-1),pred_motion_params_final[:,0])
        plt.plot(range(N_s-1),self.gt_motion_params[:,0].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x axis translation')
        plt.subplot(232)
        plt.plot(range(N_s-1),pred_motion_params_final[:,1])
        plt.plot(range(N_s-1),self.gt_motion_params[:,1].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('y axis translation')
        plt.subplot(233)
        plt.plot(range(N_s-1),pred_motion_params_final[:,2])
        plt.plot(range(N_s-1),self.gt_motion_params[:,2].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('z axis translation')
        plt.subplot(234)
        plt.plot(range(N_s-1),pred_motion_params_final[:,3])
        plt.plot(range(N_s-1),self.gt_motion_params[:,3].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x-y plane roatation')
        plt.subplot(235)
        plt.plot(range(N_s-1),pred_motion_params_final[:,4])
        plt.plot(range(N_s-1),self.gt_motion_params[:,4].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('y-z plane roatation')
        plt.subplot(236)
        plt.plot(range(N_s-1),pred_motion_params_final[:,5])
        plt.plot(range(N_s-1),self.gt_motion_params[:,5].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x-z plane roatation')
        plt.savefig(os.path.join(self.args.altopt_results_path, 'motion_pred_params_min_reconDC_loss.png'))

        if not self.args.altopt_motion_estimation_only:
            # Load the best reconstruction according to PSNR
            recon = torch.load(self.args.altopt_results_path+"/reconstruction_best_PSNR.pt")
            recon = recon.detach().cuda(self.args.gpu)
            psnr = PSNR_torch(complex_abs(recon), complex_abs(ref_img3D))
            logging.info(f"PSNR of the best reconstruction according to best PSNR: {psnr}")

            list_of_slices = None
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.altopt_results_path, "recon_axial_altOpt_best_PSNR", axis_names = ["coronal","saggital","axial"])

            # Load motion parameters from the best reconstruction according to PSNR
            pred_motion_params_final = torch.load(self.args.altopt_results_path+"/motion_params_best_PSNR.pt").numpy()
            L2_motion_parameters_final = np.sum((pred_motion_params_final - self.gt_motion_params.cpu().numpy())**2)
            logging.info(f"L2(gt_motion_params) of the best reconstruction according to best PSNR: {L2_motion_parameters_final}")
        else:
            best_L2_motion_parameters_step = self.alt_opt_meters_per_example["L2_gt_motion_parameters"].best_count
            pred_motion_params_final = self.pred_motion_params_over_epochs[:,:,best_L2_motion_parameters_step+1].numpy() # +1 as we have the all zero motion state at the beginning
            L2_motion_parameters_final = np.sum((pred_motion_params_final - self.gt_motion_params.cpu().numpy())**2)
            #assert L2_motion_parameters_final == self.alt_opt_meters_per_example["L2_gt_motion_parameters"].best_val
            logging.info(f"L2(gt_motion_params) at best L2 gt motion error for motion estimation only: {L2_motion_parameters_final}")

        
        #self.save_figure_motion_parameters_dc_loss(None, pred_motion_params_final, N_s, scale_down_factor=0.02, dc_loss_ind=-1, fig_name='motion_pred_params_best_PSNR.png')
        plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.plot(range(N_s-1),pred_motion_params_final[:,0])
        plt.plot(range(N_s-1),self.gt_motion_params[:,0].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x axis translation')
        plt.subplot(232)
        plt.plot(range(N_s-1),pred_motion_params_final[:,1])
        plt.plot(range(N_s-1),self.gt_motion_params[:,1].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('y axis translation')
        plt.subplot(233)
        plt.plot(range(N_s-1),pred_motion_params_final[:,2])
        plt.plot(range(N_s-1),self.gt_motion_params[:,2].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('z axis translation')
        plt.subplot(234)
        plt.plot(range(N_s-1),pred_motion_params_final[:,3])
        plt.plot(range(N_s-1),self.gt_motion_params[:,3].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x-y plane roatation')
        plt.subplot(235)
        plt.plot(range(N_s-1),pred_motion_params_final[:,4])
        plt.plot(range(N_s-1),self.gt_motion_params[:,4].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('y-z plane roatation')
        plt.subplot(236)
        plt.plot(range(N_s-1),pred_motion_params_final[:,5])
        plt.plot(range(N_s-1),self.gt_motion_params[:,5].cpu().numpy())
        plt.legend(['Prediction','Ground Truth'])
        plt.title('x-z plane roatation')
        plt.savefig(os.path.join(self.args.altopt_results_path, 'motion_pred_params_best_PSNR.png'))

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

    def save_figure_motion_parameters_dc_loss(self, iteration, pred_motion_params, N_s, scale_down_factor=0.02, dc_loss_ind=-1, fig_name=None):

        save_dir = os.path.join(self.args.altopt_results_path, "motion_param_figures")
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




    