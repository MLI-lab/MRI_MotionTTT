import torch
import logging
import pickle
import os
import matplotlib.pyplot as plt 

from functions.utils.helpers.meters import TrackMeter
from functions.utils.helpers.helpers_math import complex_mul, ifft2c_ndim
from functions.utils.helpers.helpers_log_save_image_utils import save_slice_images_from_volume

from functions.baselines_src.alt_opt_base_module import AltOptModuleBase
from functions.utils.data.data_loaders import invivo_loader


class AltOptModuleVivo(AltOptModuleBase):

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
        } 
            
        return alt_opt_meters_per_example

    def load_data_init_motion(self):

        masked_corrupted_kspace3D, mask3D, smaps3D_conj, traj, binary_background_mask, smaps3D = invivo_loader(self.args)
        
        self.evaluate_before_TTT(masked_corrupted_kspace3D, smaps3D_conj)

        self.gt_motion_params = None
        self.gt_traj = None
            
        return traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D
                    
    def evaluate_before_TTT(self,masked_corrupted_kspace3D, smaps3D_conj):

        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        list_of_slices = None        
        save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.altopt_results_path, "masked_corrupted_img", axis_names = ["coronal","sagittal","axial"])

    def evaluate_after_recon_step(self, recon, pred_motion_params, traj, loss_recon, loss_reg, loss_dc, iteration, recon_step, total_steps):

        self.alt_opt_meters_per_example["recon_loss_total"].update(loss_recon, total_steps)
        self.alt_opt_meters_per_example["recon_loss_dc"].update(loss_dc, total_steps)
        self.alt_opt_meters_per_example["recon_loss_reg"].update(loss_reg, total_steps)
        self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].update(loss_dc, total_steps)
                
        ### Log the data
        logging.info(f"total step {total_steps}, iteration {iteration}, recon step {recon_step} -- recon loss: {loss_recon:.5f} | DC loss: {loss_dc:.5f} | Reg loss: {loss_reg:.5f}")

        list_of_slices = None
        if iteration % 30 == 0 and recon_step == 0:
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.altopt_results_path, f"recon_total_step_{total_steps}", axis_names = ["coronal","sagittal","axial"], dir_name="recons_per_iter")


        ### Save reconstruction at minimum dc loss
        if loss_dc==self.alt_opt_meters_per_example["recon_loss_dc"].best_val:
            self.reconstruction_min_reconDC_loss = recon
            self.motion_params_min_reconDC_loss = pred_motion_params
            
        if recon_step == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.cpu(), self.args.Ns, dc_loss_ind=-1, fig_name='motion_pred_params.png')
            if iteration % 30 == 0:
                self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.cpu(), self.args.Ns, dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')

            
    def evaluate_after_motion_step(self, pred_motion_params, traj, loss_motion, iteration, motion_step, total_steps):

        self.alt_opt_meters_per_example["motion_loss"].update(loss_motion, total_steps)
        self.alt_opt_meters_per_example["dc_loss_recon_and_motion"].update(loss_motion, total_steps)

        self.pred_motion_params_over_epochs = torch.cat((self.pred_motion_params_over_epochs, pred_motion_params.cpu().unsqueeze(-1)), dim=-1)

        ### Log the data
        logging.info(f"total step {total_steps}, iteration {iteration}, motion step {motion_step} -- motion loss: {loss_motion:.5f}")


    def evaluate_after_alt_opt(self, traj):
        N_s = self.args.Ns

        torch.save(self.reconstruction_min_reconDC_loss.cpu(), self.args.altopt_results_path+"/reconstruction_min_reconDC_loss.pt")
        pickle.dump( self.alt_opt_meters_per_example, open( os.path.join(self.args.altopt_results_path, 'alt_opt_meters_per_example.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.altopt_results_path, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        best_step = self.alt_opt_meters_per_example["recon_loss_dc"].best_epoch
        best_reconDC_loss = self.alt_opt_meters_per_example["recon_loss_dc"].best_val
        ind = self.alt_opt_meters_per_example["recon_loss_dc"].best_count
        #dc_losses_per_state_norm_per_state_min_reconDC_loss = self.track_dc_losses_per_state_norm_per_state[:,ind].cpu().numpy()

        list_of_slices = None
        save_slice_images_from_volume(self.reconstruction_min_reconDC_loss[0].cpu(), list_of_slices, self.args.altopt_results_path, "recon_altOpt_min_reconDC_loss", axis_names = ["coronal","sagittal","axial"])

        # Load motion parameters from the best reconstruction according to DC loss
        pred_motion_params_final = self.motion_params_min_reconDC_loss.cpu().numpy()

        self.final_result_dict["best_reconDC_loss"] = best_reconDC_loss
        self.final_result_dict["altopt_best_reconDC_step"] = best_step 
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,ind].cpu().numpy()
        self.final_result_dict["pred_motion_params_final"] = pred_motion_params_final
        self.final_result_dict["traj"] = traj

        pickle.dump(self.final_result_dict, open( os.path.join(self.args.altopt_results_path, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        self.save_figure_motion_parameters_dc_loss(best_step, pred_motion_params_final, self.args.Ns, dc_loss_ind=ind, fig_name=f'motion_pred_params_min_reconDC_loss_totstep_{best_step}.png', )

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
        for i in range(N_s):
            plt.plot(steps_recon,self.track_dc_losses_per_state_norm_per_state[i,:].cpu().numpy(), alpha=0.1, color='blue')
        plt.xlabel('Steps')
        plt.ylabel('DC loss per state')
        plt.title('DC loss per state over steps')
        plt.savefig(self.args.altopt_results_path+"/dc_loss_convergence.png")

    def save_figure_motion_parameters_dc_loss(self, iteration, pred_motion_params, N_s, dc_loss_ind=-1, fig_name=None):

        save_dir = os.path.join(self.args.altopt_results_path, "motion_param_figures")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        titles = ["x axis translation", "y axis translation", "z axis translation", "x-y plane rotation", "y-z plane rotation", "x-z plane rotation"]

        plt.figure(figsize=(20,10))
        for i,title in enumerate(titles):
            plt.subplot(2,3,i+1)
            plt.plot(range(0,N_s,1),pred_motion_params[:,i])
            plt.ylabel('mot params in deg or mm')
            plt.xlabel('Motion states over time')
            plt.legend(['Prediction'], loc='lower left')
            plt.twinx()
            plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_per_state[:,dc_loss_ind].cpu().numpy(), 'r')
            #plt.plot(range(0,N_s,1),self.track_dc_losses_per_state_norm_per_state[:,0].cpu().numpy(), 'g')
            plt.legend([f"DC iter{dc_loss_ind}", 'DC iter0'], loc='lower right')
            plt.title(title+" iter "+str(iteration))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()





    