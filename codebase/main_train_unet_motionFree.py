
import os
import logging
import torch

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

from functions.utils.helpers.helpers_getargs import get_args

from functions.utils.helpers.helpers_init import initialize_directories, init_training

from functions.utils.data.helpers_data import get_dataloader

from functions.utils.models.helpers_model import get_model

from functions.pre_training_src.losses import get_train_loss_function
from functions.pre_training_src.unet_train_module import UnetTrainModule
from functions.baselines_src.E2E_stackedUnet_train_module import StackedUnetTrainModule


def run_train(args):

    tb_writer = init_training(args)
    logging.info(f"Start training with args: {args}")

    logging.info(f"\nRun experiment {args.experiment_name_train} with seed {args.seed} on gpu {args.gpu}.\n")

    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")
    
    torch.save(args, os.path.join(args.train_results_path, "args.pth"))

    logging.info(f"Training set {train_loader.dataset.datasets} of total length: {len(train_loader.dataset)}")
    logging.info(f"Validation set {val_loader.dataset.datasets} of total length: {len(val_loader.dataset)}")  

    # Add some more logging regarding training and validation with motion corrupted/corrected inputs
    if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs:
        logging.info(f"Training with motion simulation and motion corrected inputs {args.train_on_motion_corrected_inputs} and motion corrupted inputs {args.train_on_motion_corrupted_inputs}.")
        logging.info(f"Training motion is simulated with {args.train_max_mot} max translation/rotation, {args.train_num_motion_events} motion events, and {args.train_random_motion_seeds} random seeds per train example and motion level.")
        logging.info(f"Validation motion is simulated with {args.val_max_mot} max translation/rotation, {args.val_num_motion_events} motion events, and 1 random seed per val example and motion level.")
        logging.info(f"Further motion settings are sampling trajectory {args.sampTraj_simMot} (center in first state {args.center_in_first_state}), motion trajectory {args.motionTraj_simMot}, and number of motion states Ns {args.Ns}.")

    model, optimizer, scheduler, current_epoch = get_model(args)
    logging.info(f"We train with {args.train_batch_size_per_axis} slices per axis processed with a batch size of {args.batch_size}.")
    logging.info(f"Nufft settings are: nufft norm {args.nufft_norm}, use nufft adjoint {args.train_use_nufft_adjoint}, use nufft with dcomp {args.train_use_nufft_with_dcomp}, and nufft max coil size {args.train_nufft_max_coil_size}.")

    train_loss_function = get_train_loss_function(args)

    if args.model == "unet":
        train_module = UnetTrainModule(args, train_loader, val_loader, model, optimizer, scheduler, train_loss_function, tb_writer)
    elif args.model == "stackedUnet":
        train_module = StackedUnetTrainModule(args, train_loader, val_loader, model, optimizer, scheduler, train_loss_function, tb_writer)


    args.num_epochs = 2 if args.test_run else args.num_epochs

    if current_epoch == 0:
        train_module.val_epoch(epoch=0)
        train_module.log_after_epoch(epoch=0)

    for epoch in range(current_epoch, args.num_epochs):
        epoch += 1
        
        train_module.train_epoch(epoch)

        if epoch % args.val_every == 0:
            train_module.val_epoch(epoch)

        train_module.log_after_epoch(epoch)
        train_module.save_checkpoints(epoch, save_metrics=args.save_metrics)

    train_module.log_and_save_final_metrics(epoch)


if __name__ == '__main__':

    args = get_args()

    data_drive= "/media/ssd1/"
    path_to_data = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel"
    path_to_result_dir = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel"
    results_dir = "motion_MRI_TTT_results_tobit_kun"

    # The training code potentially allows to train on multiple datasets.
    args.train_set_paths = [
        "data_files/volume_dataset_freqEnc170_train_len40.pickle"
        ]
    args.train_data_paths = {
        'calgary-campinas-170':f"{path_to_data}/Train_converted/",
        }
    args.train_mask_paths = {
        'calgary-campinas-170':"data_files/mask_3D_size_218_170_256_R_4_poisson_disc.pickle",
        }
    args.train_sensmaps_paths = {
        'calgary-campinas-170':f"{path_to_data}/Train_s_maps_3D/",
        }
    
    args.val_set_paths = [
        "data_files/volume_dataset_freqEnc170_val_len4.pickle",
        ]
    args.val_data_paths = {
        'calgary-campinas-170':f"{path_to_data}/Val_converted/"
        }
    args.val_mask_paths = {
        'calgary-campinas-170':"data_files/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
        }
    args.val_sensmaps_paths = {
        'calgary-campinas-170':f"{path_to_data}/Val_s_maps_3D/"
        }
    
    # Set device index
    args.gpu= 0

    # # Train model
    args.train = True
    args.test_run = True

    # # Training arguments
    args.lr = 0.001
    args.num_epochs = 210
    args.scheduler = 'multistep'
    args.multistep_milestones = [180, 200]
    args.train_loss = "joint"
    args.save_metrics = ["PSNR"]
    args.num_workers = 2 
    args.val_every = 1 
    args.save_checkpoints_every = 50 
    args.log_imgs_to_tb_every = 10 
    # determines how many slices are loaded per volume in one step
    # (here we take 20 random slices from each axis)
    args.train_batch_size_per_axis = [20, 20, 20] 
    args.batch_size = 20 # batch size at which slices are processed

    # Nufft arguments
    args.nufft_norm = None
    args.train_use_nufft_adjoint = True
    args.train_use_nufft_with_dcomp = True
    args.train_nufft_max_coil_size = 12

    # # Training motion args
    args.train_on_motion_free_inputs = True
    
    # # Model arguments
    args.experiment_name_train= 'E000_Unet48_motionFree' 
    args.model= 'unet' # 'unet' or 'stackedUnet'
    args.chans= 48 # number of channels in the first layer for both unet and stackedUnet
    
    # Load a local model:
    args.load_model_path= 'None'
    args.train_load_optimizer_scheduler = False # use this option to also load optimizer and scheduler
    # Load a model from huggingface:
    args.load_model_from_huggingface = "None"

    # Sampling arguments
    args.Ns = 50 # at init number of motion states Ns must be equal to num_shots
    args.center_in_first_state = True
    args.sampTraj_simMot = "interleaved_cartesian"
    args.num_shots = 50


    # # initialize experiment, create directories
    args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")

    # # Train a model on the training set
    if args.train:
        run_train(args)








