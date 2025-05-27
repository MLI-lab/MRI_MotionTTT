
import os
from argparse import ArgumentParser
import logging
import torch

from functions.utils.helpers.helpers_getargs import get_args
from functions.utils.helpers.helpers_init import init_logging
from functions.utils.models.helpers_model import get_model
from functions.utils.data.data_loaders import invivo_loader
from functions.utils.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim
from functions.utils.motion_simulation.motion_forward_backward_models import motion_correction_NUFFT

from functions.pre_training_src.unet_train_module import UnetTrainModule
from functions.baselines_src.E2E_stackedUnet_train_module import StackedUnetTrainModule

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args(args=[])
    args.train = False
    args.train_load_optimizer_scheduler = False

    data_drive= "/media/ssd0/"

    path_to_data = "/media/ssd0/PMoC3D/dervatives/cropped_data"
    path_to_result_dir = "/media/ssd0/PMoC3D/dervatives/recon_folder"
    # Train on all levels of severity (used in paper for sim motion)
    results_dir = "stacked_unet"

    # Set device index
    args.gpu= 3

    args.nufft_norm = None
    args.train_use_nufft_adjoint= True
    args.train_use_nufft_with_dcomp= True
    args.nufft_max_coil_size= 7
    
    # # Model arguments
    args.model= 'stackedUnet' # 'unet' or 'stackedUnet'
    args.chans= 64 # number of channels in the first layer for both unet and stackedUnet
    args.norm_type_unet = "instance" # normalization for stackedUnet "batch" or "instance"
    args.in_chans = 2
    args.out_chans = 2
    args.pools = 4
    
    # Load a local model:
    args.load_model_path= 'None'
    args.load_model_from_huggingface = "mli-lab/Unet48-2D-CC359"

    model, _, _, _ = get_model(args)
    model.eval()

    # # Define train_module to access the val_step() function
    if args.model == "unet":
        train_module = UnetTrainModule(args, train_loader=None, val_loader=None, model=model, optimizer=None, scheduler=None, train_loss_function=None, tb_writer=None)
    elif args.model == "stackedUnet":
        train_module = StackedUnetTrainModule(args, train_loader=None, val_loader=None, model=model, optimizer=None, scheduler=None, train_loss_function=None, tb_writer=None)

    
    for sub_ind in range(1,9):
        for scan_ind in range(4):
            # Set data paths
            args.example_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_run-0{scan_ind}_kspace.h5')
            args.sensmaps_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_smaps.h5')
            # Give an additional name for a folder that then contains a set of experiments
            args.experiment_run_folder_name = f"vivo_{sub_ind}/"

            # # Init logging

            args.eval_unet_exp_name = "" 
            args.eval_unet_results_path = os.path.join(path_to_result_dir, results_dir, args.experiment_run_folder_name, f"S{sub_ind}_{scan_ind}{args.eval_unet_exp_name}")
            os.makedirs(args.eval_unet_results_path, exist_ok=True)
            args.eval_unet_log_path = os.path.join(args.eval_unet_results_path, "eval_unet_log.log")
            init_logging(args.eval_unet_log_path)

            logging.info(f"args: {args}")
            logging.info(f"Start end-to-end evaluation of example S{sub_ind}_{scan_ind} with model class: {args.model}")
            logging.info(f"Use local model with {sum(p.numel() for p in model.parameters()):,} parameters from {args.load_model_path} or from huggingface: {args.load_model_from_huggingface}")
            logging.info(f"Load data from {args.example_path}")
            logging.info(f"Load sensitivity maps from {args.sensmaps_path}")
            logging.info(f"Save results to {args.eval_unet_results_path}")

            args.Ns = 52
            invivo_loader_function = invivo_loader
            
            # # Load data
            masked_corrupted_kspace3D, _, smaps3D_conj, traj, binary_background_mask, _ = invivo_loader_function(args)
            
            # # Compute input image volume
            if args.train_use_nufft_adjoint:
                # Apply adjoint nufft with all zero motion parameters
                input_img_3D = motion_correction_NUFFT(masked_corrupted_kspace3D, None, traj, weight_rot=True, args=args,
                                                                            do_dcomp=args.train_use_nufft_with_dcomp, 
                                                                            num_iters_dcomp=3, max_coil_size=args.nufft_max_coil_size)
                input_img_3D = complex_mul(input_img_3D, smaps3D_conj).sum(dim=0, keepdim=False)
            else:
                input_img_3D = complex_mul(ifft2c_ndim(masked_corrupted_kspace3D, 3), smaps3D_conj).sum(dim=0, keepdim=False)
            del masked_corrupted_kspace3D, smaps3D_conj

            for ax_ind in [0,1,2]:
                logging.info(f"Reconstruct along axis {ax_ind}")
                # # Reconstruct image volume
                # move corresponding axis to batch dimension
                input_img_2D = input_img_3D.moveaxis(ax_ind, 0)
                binary_background_mask_2D = binary_background_mask[0].moveaxis(ax_ind, 0)

                with torch.no_grad():
                    recon_image_fg_1c = train_module.val_step(input_img_2D, binary_background_mask_2D,batch_size=100)
                recon_image_fg_1c = recon_image_fg_1c.moveaxis(0, ax_ind)
                del input_img_2D, binary_background_mask_2D
                                
                # # Save results
                torch.save(recon_image_fg_1c, os.path.join(args.eval_unet_results_path,f"recon_ax{ax_ind}.pt"))
                