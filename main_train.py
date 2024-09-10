
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



#torch.autograd.set_detect_anomaly(True)

def run_train(args):

    tb_writer = init_training(args)
    logging.info(f"Start training with args: {args}")

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu}.\n")

    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")
    
    torch.save(args, os.path.join(args.train_results_path, "args.pth"))

    logging.info(f"Training set {train_loader.dataset.datasets} of total length: {len(train_loader.dataset)}")
    logging.info(f"Validation set {val_loader.dataset.datasets} of total length: {len(val_loader.dataset)}")    

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

    args.train_set_paths = [
        "volume_dataset_freqEnc170_180_train_len47.pickle"
        ]
    args.train_data_paths = {
        'calgary-campinas-170':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/",
        'calgary-campinas-180':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/"
        }
    args.train_mask_paths = {
        'calgary-campinas-170':"mask_3D_size_218_170_256_R_4_poisson_disc.pickle",
        'calgary-campinas-180':"mask_3D_size_218_180_256_R_4_poisson_disc.pickle"
        }
    args.train_sensmaps_paths = {
        'calgary-campinas-170':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_s_maps_3D/",
        'calgary-campinas-180':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_s_maps_3D/"
        }
    
    args.val_set_paths = [
        "volume_dataset_freqEnc170_val_len4.pickle"
        ]
    args.val_data_paths = {
        'calgary-campinas-170':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/"
        }
    args.val_mask_paths = {
        'calgary-campinas-170':"mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
        }
    args.val_sensmaps_paths = {
        'calgary-campinas-170':"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_s_maps_3D/"
        }
    
    args.data_drive= "/media/ssd3/"
    args.gpu= 1

    args.train = True

    args.nufft_norm = None

    # # Training arguments
    args.train_use_nufft_adjoint = True
    args.train_use_nufft_with_dcomp = True
    args.train_on_motion_corrected_inputs = False # True currently not supported
    args.train_on_motion_corrupted_inputs = False # True currently not supported
    args.train_one_grad_step_per_image_in_batch = False
    args.train_only_first_last_layers = False
    args.lr = 0.001
    args.num_epochs = 240
    args.scheduler = 'multistep'
    args.multistep_milestones = [200, 220]
    args.train_loss = "joint"
    args.save_metrics = ["PSNR"]
    args.num_workers = 2 # !!
    args.val_every = 2
    args.save_checkpoints_every = 50 # !!
    args.log_imgs_to_tb_every = 10 # !!
    args.train_batch_size_per_axis = [20, 20, 20]
    args.train_max_rots = [2, 4, 6, 8, 10]
    args.train_max_trans = [2, 4, 6, 8, 10]
    args.train_Ns = [5,10,20]
    args.train_num_random_motion_seeds = 1000

    # # Model arguments
    args.experiment_name= 'E028_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp_CalgaryFull'# 'E015_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp'
    args.TTT_from_which_checkpoint= 'best_PSNR'
    args.model= 'unet'
    args.chans= 48
    args.load_1ch_model = False
    args.train_load_optimizer_scheduler = False
    args.load_external_model_path = "None"
    #None 
    #"/tobit/revisit_TTT/E080_unet_brain20000_joint_SENSE/checkpoints/checkpoint_best_PSNR_sense.pt"

    # # motion arguments (not needed for model TTT)
    args.Ns = 1 # important to set to 1 for unet training
    args.num_motion_events = 5
    args.max_trans = 2
    args.max_rot = 2
    args.random_motion_seed = 57

    # initialize experiment, create directories
    args = initialize_directories(args)

    if args.train:
        run_train(args)







