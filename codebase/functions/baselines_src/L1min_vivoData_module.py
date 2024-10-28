
import torch
import logging
import pickle
import os
import matplotlib.pyplot as plt 

from functions.utils.helpers.meters import TrackMeter
from functions.utils.helpers.helpers_math import complex_mul, ifft2c_ndim
from functions.utils.helpers.helpers_log_save_image_utils import save_slice_images_from_volume

from functions.baselines_src.L1min_base_module import L1minModuleBase
from functions.utils.data.data_loaders import invivo_loader


class L1minModuleVivo(L1minModuleBase):

    def __init__(
            self,
            args,
            ) -> None:
        super().__init__(args)

        self.L1min_meters = self.init_L1min_meters()

    def init_L1min_meters(self):
        #  Decide what quantities to track during L1 min
        L1min_meters = {
            "loss_total" : TrackMeter('decaying'),    
            "loss_dc" : TrackMeter('decaying'),
            "loss_reg" : TrackMeter('decaying'),
        } 
            
        return L1min_meters

    def load_data_init_motion(self):

        masked_corrupted_kspace3D, mask3D, smaps3D_conj, traj, binary_background_mask, smaps3D = invivo_loader(self.args)
        
        self.evaluate_before_L1min(masked_corrupted_kspace3D, smaps3D_conj)

        self.gt_motion_params = None
        self.gt_traj = None
            
        return traj, smaps3D, binary_background_mask, masked_corrupted_kspace3D, mask3D
                    
    def evaluate_before_L1min(self,masked_corrupted_kspace3D, smaps3D_conj):

        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        list_of_slices = None        
        save_slice_images_from_volume(masked_corrupted_img3D.cpu(), list_of_slices, self.args.L1min_results_path, "masked_corrupted_img", axis_names = ["coronal","sagittal","axial"])


    def evaluate_after_L1min_step(self, recon, pred_motion_params, traj, loss_recon, loss_reg, loss_dc, iteration):

        self.L1min_meters["loss_total"].update(loss_recon, iteration)
        self.L1min_meters["loss_dc"].update(loss_dc, iteration)
        self.L1min_meters["loss_reg"].update(loss_reg, iteration)
                
        ### Log the data
        logging.info(f"iteration {iteration} -- recon loss: {loss_recon:.5f} | DC loss: {loss_dc:.5f} | Reg loss: {loss_reg:.5f}")

        list_of_slices = None
        if iteration % 15 == 0:
            save_slice_images_from_volume(recon[0].cpu(), list_of_slices, self.args.L1min_results_path, f"recon_total_step_{iteration}", axis_names = ["coronal","sagittal","axial"], dir_name="recons_per_iter")

        ### Save reconstruction at minimum dc loss
        if loss_dc==self.L1min_meters["loss_dc"].best_val:
            self.reconstruction_min_reconDC_loss = recon
            
        if iteration % 15 == 0:
            self.save_figure_motion_parameters_dc_loss(iteration, pred_motion_params.cpu(), self.args.Ns, dc_loss_ind=-1, fig_name=f'motion_pred_params_{iteration}.png')

            
    def evaluate_after_L1min(self, traj, pred_motion_params):
        N_s = self.args.Ns

        torch.save(self.reconstruction_min_reconDC_loss.cpu(), self.args.L1min_results_path+"/reconstruction_min_reconDC_loss.pt")
        pickle.dump( self.L1min_meters, open( os.path.join(self.args.L1min_results_path, 'L1min_meters.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_per_state.cpu().numpy(), open( os.path.join(self.args.L1min_results_path, 'track_dc_losses_per_state_norm_per_state.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump(self.track_dc_losses_per_state_norm_all_states.cpu().numpy(), open( os.path.join(self.args.L1min_results_path, 'track_dc_losses_per_state_norm_all_states.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        best_step = self.L1min_meters["loss_dc"].best_epoch
        best_reconDC_loss = self.L1min_meters["loss_dc"].best_val

        list_of_slices = None
        save_slice_images_from_volume(self.reconstruction_min_reconDC_loss[0].cpu(), list_of_slices, self.args.L1min_results_path, "recon_min_reconDC_loss", axis_names = ["coronal","sagittal","axial"])

        self.final_result_dict["best_reconDC_loss"] = best_reconDC_loss
        self.final_result_dict["best_reconDC_step"] = best_step 
        self.final_result_dict["dc_losses_per_state_norm_per_state_min_reconDC_loss"] = self.track_dc_losses_per_state_norm_per_state[:,best_step].cpu().numpy()
        self.final_result_dict["pred_motion_params"] = pred_motion_params
        self.final_result_dict["traj"] = traj

        pickle.dump(self.final_result_dict, open( os.path.join(self.args.L1min_results_path, 'final_result_dict.pkl'), "wb" ) , pickle.HIGHEST_PROTOCOL )

        self.save_figure_motion_parameters_dc_loss(best_step, pred_motion_params.cpu().numpy(), self.args.Ns, dc_loss_ind=best_step, fig_name=f'motion_pred_params_min_reconDC_loss_totstep_{best_step}.png', )

        DC_losses = self.L1min_meters["loss_dc"].val
        steps = self.L1min_meters["loss_dc"].epochs
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(steps,DC_losses)
        plt.xlabel('Steps')
        plt.ylabel('DC loss')
        plt.title('DC loss over steps')
        plt.subplot(1,2,2)
        for i in range(N_s):
            plt.plot(steps,self.track_dc_losses_per_state_norm_per_state[i,:].cpu().numpy(), alpha=0.1, color='blue')
        plt.xlabel('Steps')
        plt.ylabel('DC loss per state')
        plt.title('DC loss per state over steps')
        plt.savefig(self.args.L1min_results_path+"/dc_loss_convergence.png")

    def save_figure_motion_parameters_dc_loss(self, iteration, pred_motion_params, N_s, dc_loss_ind=-1, fig_name=None):

        save_dir = os.path.join(self.args.L1min_results_path, "motion_param_figures")
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
            plt.legend([f"DC iter{dc_loss_ind}", 'DC iter0'], loc='lower right')
            plt.title(title+" iter "+str(iteration))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        if fig_name is not None:
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.savefig(os.path.join(save_dir, 'motion_pred_params.png'))
        plt.close()



