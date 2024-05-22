
import os
import logging
import torch
import pickle

from functions.helpers.helpers_getargs import get_args

from functions.helpers.helpers_init import initialize_directories, init_TTT, init_alt_opt

from functions.models.helpers_model import get_model

from functions.test_time_training.unet_TTT_module import UnetTTTModule
from functions.test_time_training.alt_opt_module import AltOptModule


def run_TTT(args):

    tb_writer = init_TTT(args)
    logging.info(f"Start TTT with args: {args}")

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu} with motion TTT lr {args.lr_TTT}.\n")
    

    args.load_model_from_checkpoint = f"checkpoint_{args.TTT_from_which_checkpoint}.pt"

    torch.save(args, os.path.join(args.TTT_results_path, "args.pth"))
    with open(os.path.join(args.TTT_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    logging.info(f"Apply TTT to example from {args.TTT_example_path}")
    logging.info(f"Load sensitivity maps from {args.TTT_sensmaps_path}")

    model, _, _ = get_model(args)

    if args.model == "unet":
        TTT_module = UnetTTTModule(args, model, tb_writer)
        logging.info("Starting simulated motion TTT")

    args.num_steps_TTT = 2 if args.test_run else args.num_steps_TTT
    
    # !!!!
    # Implement TTT for motion correction.
    TTT_module.TTT()

def run_alt_opt(args):

    init_alt_opt(args)

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu}.\n")
    logging.info(f"Apply alternating optimization to example from {args.TTT_example_path}")
    logging.info(f"Alt opt experiment: {args.experiment_name_alt_opt}")

    torch.save(args, os.path.join(args.altopt_results_path, "args.pth"))
    with open(os.path.join(args.altopt_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    alt_opt_module = AltOptModule(args)
    logging.info("Starting alternating optimization recon")

    alt_opt_module.alt_opt()



if __name__ == '__main__':

    args = get_args()
    args.gpu= 1
    args.alt_opt=True # run in recon_only mode to get L1 recon after TTT
    args.TTT=True
    args.train = False

    # # Load either list of test examples of list of validation examples
    args.data_drive= "/media/ssd0/" # prefix for all paths
    args.save_exp_results_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/motion_MRI_TTT_results_tobit_kun/"
    args.finalTestset_data_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/volume_dataset_freqEnc170_test_len5_1.pickle"
    args.val_set_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/volume_dataset_freqEnc170_val_len4.pickle"

    with open(os.path.join(args.data_drive, args.finalTestset_data_path),'rb') as fn:
        examples_list_of_dicts_test = pickle.load(fn)
    with open(os.path.join(args.data_drive, args.val_set_path),'rb') as fn:
        examples_list_of_dicts_val = pickle.load(fn)

    filename = examples_list_of_dicts_test[0]["filename"] 
    # For test set we used seed1 and seed2 to generate motion. For val set set seed to 0 or 1
    random_motion_seed = examples_list_of_dicts_test[0]['motion_seeds']['seed1']

    # Paths for running MotionTTT or AltOpt
    args.TTT_example_path = f"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/{filename}"
    args.TTT_mask_path= "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
    args.TTT_sensmaps_path= "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_s_maps_3D/"

    # Set to save TTT/AltOpt results to a specific folder
    args.finalTestset = False
    args.experiment_run_folder_name = "check_code_submission/"

    # # Model arguments
    args.experiment_name= 'E025_code_submission'
    args.TTT_from_which_checkpoint= 'best_PSNR'
    args.model= 'unet'
    args.chans= 48
    args.load_external_model_path = "None"

    # # motion TTT arguments 
    args.experiment_name_TTT= "_settings_from_paper"
    args.lr_TTT= 1.0
    args.TTT_optimizer = 'Adam'
    args.num_steps_TTT= 150
    args.num_slices_per_grad_step= 5
    args.TTT_all_axes= True
    args.TTT_use_nufft_with_dcomp = True
    args.TTT_nufft_max_coil_size = 4 # None for full batch
    args.TTT_num_rot_only_grad_steps = 5 
    args.TTT_use_clamp_schedule = True
    args.TTT_only_motCorrupt_grad = False
    args.TTT_motState_batchSize_per_backprop = 25


    # # alternating optimization arguments on top of a TTT experiment
    args.alt_opt_on_TTTexp = True # if True then set args.alt_opt = True

    # # alternating optimization arguments
    args.altopt_recon_only=True
    args.altopt_dc_thresholding = True
    args.altopt_dc_threshold = 0.65
    #if true estimated motion parameters are aligned with true motion parameters before L1-minimization
    args.altopt_align_motParams = True
    args.altopt_motion_estimation_only=False
    args.altopt_recon_only_with_motionKnowledge = False
    args.experiment_name_alt_opt= f"_settings_from_paper"
    args.altopt_steps_total = 1
    args.altopt_nufft_max_coil_size = 6
    
    args.altopt_steps_recon = 50
    args.altopt_lr_recon = 5e7 # 5e7 for train data
    args.altopt_lam_recon = 1e-3 #1e-3 for train data
    args.altopt_optimizer_recon = 'SGD'

    args.altopt_steps_motion = 0
    args.altopt_lr_motion = 5e-11
    args.altopt_optimizer_motion = 'SGD'

    # # motion arguments
    args.Ns = 50
    args.num_motion_events = 5
    args.max_trans = 2
    args.max_rot = 2
    args.random_motion_seed = random_motion_seed

    # initialize experiment, create directories
    args = initialize_directories(args)

    if args.TTT:
        run_TTT(args)

    if args.alt_opt:
        run_alt_opt(args)





