
import os
import logging
import torch
import pickle
import copy

from functions.helpers.helpers_getargs import get_args

from functions.helpers.helpers_init import initialize_directories_TTT, init_TTT, init_alt_opt
from functions.models.helpers_model import get_model

from functions.test_time_training.unet_TTT_module import UnetTTTModule
from functions.test_time_training.alt_opt_module import AltOptModule
from functions.test_time_training.unet_TTT_vivo import UnetTTTModuleVivo
from functions.test_time_training.alt_opt_vivo import AltOptModuleVivo


def run_TTT(args):

    init_TTT(args)
    logging.info(f"Start TTT with args: {args}")

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu}.\n")
    

    args.load_model_from_checkpoint = f"checkpoint_{args.TTT_from_which_checkpoint}.pt"

    with open(os.path.join(args.TTT_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    logging.info(f"Apply TTT to example from {args.TTT_example_path}")
    logging.info(f"Load sensitivity maps from {args.TTT_sensmaps_path}")

    model, _, _ = get_model(args)

    if args.model == "unet" and args.TTT_vivo:
        TTT_module = UnetTTTModuleVivo(args, model)
        logging.info("Starting in vivo TTT")

    elif args.model == "unet":
        TTT_module = UnetTTTModule(args, model)
        logging.info("Starting simulated motion TTT")

    TTT_module.run_TTT()

    if args.TTT_finetune_after_DCTh:
        TTT_module.finetune_after_DCTh()

def run_alt_opt(args):

    init_alt_opt(args)

    logging.info(f"\nRun experiment {args.experiment_name} with seed {args.seed} on gpu {args.gpu}.\n")
    logging.info(f"Apply alternating optimization to example from {args.TTT_example_path}")
    logging.info(f"Alt opt experiment: {args.experiment_name_alt_opt}")

    torch.save(args, os.path.join(args.altopt_results_path, "args.pth"))
    with open(os.path.join(args.altopt_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    if args.alt_opt_vivo:
        alt_opt_module = AltOptModuleVivo(args)
        logging.info("Starting alternating optimization recon on in vivo data")
    else:
        alt_opt_module = AltOptModule(args)
        logging.info("Starting alternating optimization recon")

    alt_opt_module.run_alt_opt()


if __name__ == '__main__':

    args = get_args()

    for example in [
        "ri_06082024_1740087_7_2_wip_rand_cs4rad_t1w_3d_tfeV4",
        "ri_06082024_1747120_9_2_wip_rand_cs4rad_t1w_3d_tfeV4",
        "ri_06082024_1750291_10_2_wip_rand_cs4rad_t1w_3d_tfeV4",
        "ri_06082024_1736103_6_2_wip_rand_cs4rad_t1w_3d_tfeV4",
        "ri_06082024_1743469_8_2_wip_rand_cs4rad_t1w_3d_tfeV4",]:
        
        #args.TTT_example_path = f"motion_corrupted_stefan/moco_data0/{example}.mat.h5"
        args.TTT_example_path = f"motion_corrupted_stefan/moco_data_randTraj/{example}.mat.h5"

        #args.TTT_example_path = "motion_corrupted_stefan/moco_data0/mo_19042024_2206106_10_2_wip_cs4_t1w_3d_tfeV4.mat.h5"

        #"motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1736103_6_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1743469_8_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1750291_10_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1740087_7_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1747120_9_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"

        #"motion_corrupted_stefan/moco_data0/mo_19042024_2141076_4_2_wip_cs4_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2144093_5_2_wip_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2146581_6_2_wip_cs4lowhigh_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2150515_7_2_wip_cs4_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2153414_8_2_wip_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2156305_9_2_wip_cs4lowhigh_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2206106_10_2_wip_cs4_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2209004_11_2_wip_cs4rad_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2211487_12_2_wip_cs4lowhigh_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2223449_14_2_wip_cs4_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2230499_16_2_wip_cs4lowhigh_t1w_3d_tfeV4.mat.h5"
        #"motion_corrupted_stefan/moco_data0/mo_19042024_2234019_17_2_wip_cs4rad_t1w_3d_tfeV4.mat.h5"


        #args.TTT_sensmaps_path = "motion_corrupted_stefan/moco_data0/mo_19042024_2141076_4_2_wip_cs4_t1w_3d_tfeV4.mat.h5"
        args.TTT_sensmaps_path = "motion_corrupted_stefan/moco_data_randTraj/ri_06082024_1736103_6_2_wip_rand_cs4rad_t1w_3d_tfeV4.mat.h5"

        args.data_drive= "/media/ssd3/"
        args.gpu= 0
        args.alt_opt=True
        args.alt_opt_vivo=True #alt_opt also needs to be True
        args.TTT=True
        args.TTT_vivo=True #TTT also needs to be True

        args.finalTestset = False

        # Give an additional name for a folder that then contains a set of experiments
        args.experiment_run_folder_name = "vivo_randTraj/"

        # Nufft arguments
        args.nufft_norm = None
        args.TTT_use_nufft_with_dcomp = True

        # # Model arguments
        args.experiment_name= 'E015_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp'# 'E015_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp'
        args.TTT_from_which_checkpoint= 'best_PSNR'
        args.model= 'unet'
        args.chans= 48
        args.load_1ch_model = False
        args.train_load_optimizer_scheduler = False
        args.load_external_model_path = "None"

        # # motion TTT arguments
        args.experiment_name_TTT= "_3phase_v1_th0.70_splits10"

        # Optimization arguments
        args.lr_TTT= 1.0
        args.TTT_optimizer = 'Adam'
        args.TTT_lr_max_decays = 2
        args.TTT_lr_decay_factor = 0.25
        args.TTT_lr_decay_after = 400
        args.TTT_lr_decay_at_the_latest = [49,1000]
        args.num_steps_TTT= 130
        args.TTT_set_DCloss_lr = False
        args.TTT_set_DCloss_lr_th = 0.70
        args.TTT_all_axes= True
        args.TTT_num_rot_only_grad_steps = 5
        args.TTT_use_clamp_schedule = True
        args.TTT_norm_per_shot = False # change loss normalization

        # Control GPU memory consumption and computational time
        args.num_slices_per_grad_step= 5
        args.TTT_motState_batchSize_per_backprop = 8 # None for full batch
        args.TTT_nufft_max_coil_size = 3 # None for full batch
        args.TTT_only_motCorrupt_grad = False
        if args.TTT_only_motCorrupt_grad:
            raise ValueError("TTT_only_motCorrupt_grad implementation outdated. Please set to False.")
        
        # Specify iterations at which states detected by TTT_set_DCloss_lr_th are reset to 0
        # Currently after reset the optimizer is reset and we contine to optimize over all states
        args.TTT_list_of_reset_steps = []

        # Specify iterations at which states detected by TTT_set_DCloss_lr_th are split
        # into TTT_states_per_split states. If TTT_all_states_grad_after_split is False
        # then only the states that are split are optimized over for the remainin iterations
        args.TTT_list_of_split_steps = [71]
        args.TTT_states_per_split = 10
        args.TTT_all_states_grad_after_split = False
        args.TTT_lr_after_split = 0.5
        args.TTT_DCloss_th_split = 0.64

        # Set an interation after which motionTTT defines a new optimizer and starts optimizing
        # over all motion states in the current pred_motion_params for the remaining iterations
        args.TTT_optimize_all_states_after = 100
        args.TTT_optimize_all_states_after_lr = 0.05

        if args.TTT_list_of_split_steps != [] and args.TTT_optimize_all_states_after is not None:
            assert args.TTT_optimize_all_states_after > args.TTT_list_of_split_steps[-1]

        # Start MotionTTT from a given set of motion parameters
        args.TTT_path_to_pred_motion_params = None
        if args.TTT_path_to_pred_motion_params is not None:
            raise ValueError("TTT_path_to_pred_motion_params implementation has not been tested. Please set to None.")

        # # motion TTT fine tune after DC loss thresholding arguments
        # Load motion parameters from MotionTTT, apply DC loss thresholding and fine tune
        # the remaining states.
        args.TTT_finetune_after_DCTh = False
        if args.TTT_finetune_after_DCTh:
            raise ValueError("TTT_finetune_after_DCTh implementation has not been tested together with TTT_list_of_split_steps. Please set to False.")
        args.lr_TTT_finetune = 1e-1
        args.num_steps_TTT_finetune = 50
        args.experiment_name_TTT_finetune= f"steps{args.num_steps_TTT_finetune}_lr{args.lr_TTT_finetune:.1e}"

        # # alternating optimization arguments on top of a TTT experiment
        args.alt_opt_on_TTTexp = True # if True then set args.alt_opt = True
        args.alt_opt_on_TTT_load_from_phase = 0 # if splitting was used set phase=1 to recon results after split

        # # alternating optimization arguments
        args.altopt_recon_only=True
        args.altopt_dc_thresholding = True
        args.altopt_dc_threshold = 0.70
        args.experiment_name_alt_opt= f"_dcTh_{args.altopt_dc_thresholding}_{args.altopt_dc_threshold}"
        args.altopt_steps_total = 50
        args.altopt_nufft_max_coil_size = 7
        
        args.altopt_steps_recon = 1
        args.altopt_lr_recon = 1e8 # 5e7 for train data, 1e8 for scan data
        args.altopt_lam_recon = 3e-8 #1e-3 for train data, 3e-8 for scan data
        args.altopt_optimizer_recon = 'SGD'

        # # motion arguments
        args.Ns = 52 # SET Ns also below!!!
        args.center_in_first_state = False
        args.fix_mot_maxksp_shot = False

        # initialize experiment, create directories
        args = initialize_directories_TTT(args, TTT_fintune=False)

        if args.TTT:
            run_TTT(args)

        if args.alt_opt:
            args.Ns = 52
            args = initialize_directories_TTT(args, TTT_fintune=False)
            run_alt_opt(args)

            # Always obtain recon without DC thresholding
            if args.altopt_dc_thresholding:
                args.Ns = 52
                args.altopt_dc_thresholding = False
                args.experiment_name_alt_opt= f"_dcTh_{args.altopt_dc_thresholding}_{args.altopt_dc_threshold}"
                args = initialize_directories_TTT(args, TTT_fintune=False)
                run_alt_opt(args)

        if args.TTT_finetune_after_DCTh:
            args.Ns = 52
            args = initialize_directories_TTT(args, TTT_fintune=True)
            motion_TTT_results_path = os.path.join(args.TTT_results_path_numerical, 'final_result_dict.pkl')
            if os.path.exists(motion_TTT_results_path):
                run_alt_opt(args)





