
import os
import logging
import torch
import pickle

from functions.helpers.helpers_getargs import get_args

from functions.helpers.helpers_init import initialize_directories, init_training

from functions.data.helpers_data import get_dataloader

from functions.models.helpers_model import get_model

from functions.training.losses import get_train_loss_function
from functions.training.unet_train_module import UnetTrainModule



def run_train(args):
    
    tb_writer = init_training(args)

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu}.\n")

    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")

    logging.info(f"Start training with args: {args}")
    
    torch.save(args, os.path.join(args.train_results_path, "args.pth"))
    with open(os.path.join(args.train_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    logging.info(f"Training set {train_loader.dataset.dataset} of length: {len(train_loader.dataset)}")
    logging.info(f"Validation set {val_loader.dataset.dataset} of length: {len(val_loader.dataset)}")    

    model, optimizer, scheduler = get_model(args)

    

    train_loss_function = get_train_loss_function(args)

    if args.model == "unet":
        train_module = UnetTrainModule(args, train_loader, val_loader, model, optimizer, scheduler, train_loss_function, tb_writer)


    args.num_epochs = 2 if args.test_run else args.num_epochs

    if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint:
        train_module.val_epoch_volume(epoch=0)
    else:
        train_module.val_epoch(epoch=0)

    train_module.log_after_epoch(epoch=0)

    for epoch in range(args.num_epochs):
        epoch += 1
        
        if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint:
            train_module.train_epoch_volume(epoch)
        else:
            train_module.train_epoch(epoch)

        if epoch % args.val_every == 0:
            if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint:
                train_module.val_epoch_volume(epoch)
            else:
                train_module.val_epoch(epoch)

        train_module.log_after_epoch(epoch)
        train_module.save_checkpoints(epoch, save_metrics=args.save_metrics)

    train_module.log_and_save_final_metrics(epoch)


if __name__ == '__main__':

    args = get_args()
    args.gpu= 1
    args.alt_opt=False
    args.TTT=False
    args.train = True

    # # Load either list of test examples of list of validation examples
    args.data_drive= "/media/ssd0/" # prefix for all paths
    args.save_exp_results_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/motion_MRI_TTT_results_tobit_kun/"

    # Paths for training
    args.train_set_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/volume_dataset_freqEnc170_train_len40.pickle"
    args.train_data_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/"
    args.train_mask_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
    args.train_sensmaps_path="cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_s_maps_3D/"
    args.val_set_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/volume_dataset_freqEnc170_val_len4.pickle"
    args.val_data_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/"
    args.val_mask_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
    args.val_sensmaps_path="cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_s_maps_3D/"

    # # Training arguments
    args.train_use_nufft_adjoint = True
    args.train_use_nufft_with_dcomp = True
    args.train_on_motion_corrected_inputs = False
    args.train_on_motion_corrupted_inputs = False
    args.train_one_grad_step_per_image_in_batch = False
    args.lr = 0.001
    args.num_epochs = 240
    args.scheduler = 'multistep'
    args.multistep_milestones = [200]
    args.train_loss = "joint"
    args.save_metrics = ["PSNR"]
    args.num_workers = 4 # !!
    args.val_every = 1
    args.save_checkpoints_every = 20 # !!
    args.log_imgs_to_tb_every = 5 # !!
    args.train_batch_size_per_axis = [20, 20, 20]

    # # Model arguments
    args.experiment_name= 'E025_code_submission'#'E015_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp'
    args.TTT_from_which_checkpoint= 'best_PSNR'
    args.model= 'unet'
    args.chans= 48
    args.load_external_model_path = "None"

    # # motion arguments
    args.Ns = 1

    # initialize experiment, create directories
    args = initialize_directories(args)

    if args.train:
        run_train(args)





