
import os
import logging
import torch
import pickle
import traceback
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

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

    args.data_drive= "/media/ssd3/"

    args.finalTestset_data_path = "volume_dataset_freqEnc170_test_len5_1.pickle"
    #args.finalTestset_data_path_2 = "volume_dataset_freqEnc170_test_len5_2.pickle"
    #args.valset_data_path = "volume_dataset_freqEnc170_val_len4.pickle"

    with open(os.path.join(args.data_drive, args.finalTestset_data_path),'rb') as fn:
        examples_list_of_dicts = pickle.load(fn)

    for Ns in [50]:
        for TTT_num_shots in [50]:
            for TTT_states_per_split in [10]:
                for max_mot in [10,5,2]:  
                    for num_motion_events in [10,5,1]:
                        for example_dict in examples_list_of_dicts:
                            filename = example_dict["filename"]  
                        
                            for random_motion_seed_ind in ['seed1','seed2']:
                                random_motion_seed = example_dict['motion_seeds'][random_motion_seed_ind]

                                #args.TTT_example_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/e14583s3_P21504.7.h5"
                                
                                args.TTT_example_path = f"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/{filename}"
                                
                                args.TTT_mask_path= "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
                                args.gpu= 0
                                args.alt_opt=True
                                args.alt_opt_vivo=False #alt_opt also needs to be True
                                args.TTT=True
                                args.TTT_vivo=False #TTT also needs to be True

                                args.finalTestset = False

                                args.seed = 1

                                # Give an additional name for a folder that then contains a set of experiments
                                args.experiment_run_folder_name = "test_intraShot_randTraj_fixFirstState_initlr4.0_3phase_splits10_initSplitMotAvg_run54/"

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
                                
                                # Optimization arguments
                                args.lr_TTT= 4.0 # only relevant if TTT_set_DCloss_lr=False
                                args.TTT_optimizer = 'Adam'
                                args.TTT_lr_max_decays = 3 # to 1.0, 0.25, 0.0625
                                args.TTT_lr_decay_factor = 0.25
                                args.TTT_lr_decay_after = 400 # not used
                                args.TTT_lr_decay_at_the_latest = [39, 59, 1000]
                                args.num_steps_TTT= 130
                                args.TTT_set_DCloss_lr = False
                                args.TTT_set_DCloss_lr_th = 0.45
                                args.TTT_all_axes= True
                                args.TTT_num_rot_only_grad_steps = 5
                                args.TTT_use_clamp_schedule = True
                                args.TTT_norm_per_shot = False # change loss normalization

                                # Control GPU memory consumption and computational time
                                args.num_slices_per_grad_step= 5
                                args.TTT_motState_batchSize_per_backprop = 25 # None for full batch
                                args.TTT_nufft_max_coil_size = 6 # None for full batch
                                args.TTT_only_motCorrupt_grad = False
                                if args.TTT_only_motCorrupt_grad:
                                    raise ValueError("TTT_only_motCorrupt_grad implementation outdated. Please set to False.")
                                
                                # Specify iterations at which states detected by TTT_set_DCloss_lr_th are reset to 0
                                # Currently after reset the optimizer is reset and we contine to optimize over all states
                                args.TTT_list_of_reset_steps = []

                                # If TTT_intraShot_estimation_only is True then in the first iteration the predicted motion
                                # parameters are set to ground truth values except for shots at which intra-shot motion is occurs.
                                # For those shots the predicted motion parameters are split according to TTT_states_per_split and initilized with zeros.
                                args.TTT_intraShot_estimation_only = False

                                # Specify iterations at which states detected by TTT_set_DCloss_lr_th are split
                                # into TTT_states_per_split states. If TTT_all_states_grad_after_split is False
                                # then only the states that are split are optimized over for the remainin iterations
                                args.TTT_list_of_split_steps = [71]
                                args.TTT_states_per_split = TTT_states_per_split
                                args.TTT_all_states_grad_after_split = False
                                args.TTT_lr_after_split = 0.5
                                args.TTT_DCloss_th_split = 0.575

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
                                args.TTT_finetune_after_DCTh = False
                                if args.TTT_finetune_after_DCTh:
                                    raise ValueError("TTT_finetune_after_DCTh implementation outdated. Please set to False.")
                                args.lr_TTT_finetune = 1e-1
                                args.num_steps_TTT_finetune = 50
                                args.experiment_name_TTT_finetune= f"steps{args.num_steps_TTT_finetune}_lr{args.lr_TTT_finetune:.1e}"

                                # # alternating optimization arguments on top of a TTT experiment
                                args.alt_opt_on_TTTexp = True # if True then set args.alt_opt = True
                                args.alt_opt_on_TTT_load_from_phase = 0 # if splitting was used set phase=1 to recon results after split
                                
                                # # alternating optimization arguments
                                args.altopt_recon_only=True
                                args.altopt_dc_thresholding = True
                                args.altopt_dc_threshold = 0.575
                                args.altopt_align_motParams = False # only important for simulated motion
                                args.altopt_motion_estimation_only = False
                                args.altopt_recon_only_with_motionKnowledge = False
                                args.experiment_name_alt_opt= "placeholder"
                                args.altopt_steps_total = 50
                                args.altopt_nufft_max_coil_size = 12#6
                                
                                args.altopt_steps_recon = 1
                                args.altopt_lr_recon = 5e7 # 5e7 for train data, 1e8 for scan data
                                args.altopt_lam_recon = 1e-3 #1e-3 for train data, 3e-8 for scan data
                                args.altopt_optimizer_recon = 'SGD'

                                args.altopt_steps_motion = 0

                                # # Motion and sampling trajectory arguments
                                # Motion trajectory choices: 'uniform_interShot_event_model', 'uniform_intraShot_event_model'
                                args.Ns = Ns # currently must be equal to TTT_num_shots
                                args.center_in_first_state = True
                                args.fix_mot_maxksp_shot = True
                                args.motionTraj_simMot = "uniform_intraShot_event_model" 
                                args.TTT_num_shots = TTT_num_shots 
                                args.num_motion_events = num_motion_events
                                args.num_intraShot_events = int(np.ceil(num_motion_events/2))
                                args.max_trans = max_mot
                                args.max_rot = max_mot
                                args.random_motion_seed = random_motion_seed

                                # Sampling trajectory choices: 
                                # 'interleaved_cartesian_Ns500', 'random_cartesian', 'deterministic_cartesian', 'interleaved_cartesian', 'linear_cartesian
                                args.TTT_sampTraj_simMot = "random_cartesian"
                                args.random_sampTraj_seed = 0 # for random_cartesian
                                args.sampling_order_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc_traj_DS5_MODEquadruples_center3_order03142.pickle"


                                args.experiment_name_TTT= f"_rottrans{max_mot}_Ns{Ns}_Nshots{TTT_num_shots}_motEvents{num_motion_events}_motSeed{random_motion_seed}"

                                # initialize experiment, create directories
                                args = initialize_directories_TTT(args, TTT_fintune=False)

                                if args.TTT:
                                    run_TTT(args)

                                if args.alt_opt:
                                    for altopt_dc_thresholding in [True, False]:
                                        for altopt_align_motParams in [True, False]:
                                            for phase in [0,1,2]:
                                                try:                                                
                                                    args.Ns = Ns
                                                    args.alt_opt_on_TTT_load_from_phase = phase
                                                    args.altopt_dc_thresholding = altopt_dc_thresholding
                                                    args.altopt_align_motParams = altopt_align_motParams
                                                    args.experiment_name_alt_opt= f"phase{args.alt_opt_on_TTT_load_from_phase}_dcTh{args.altopt_dc_threshold}_{args.altopt_dc_thresholding}_align_{args.altopt_align_motParams}"
                                                    args = initialize_directories_TTT(args, TTT_fintune=False)
                                                    run_alt_opt(args)
                                                except:
                                                    logging.info(f"Phase {phase} failed due to:")
                                                    error_str = traceback.format_exc()
                                                    logging.info(error_str)  
                                                    pass

                                if args.TTT_finetune_after_DCTh:
                                    args.Ns = Ns
                                    args = initialize_directories_TTT(args, TTT_fintune=True)
                                    motion_TTT_results_path = os.path.join(args.TTT_results_path_numerical, 'final_result_dict.pkl')
                                    if os.path.exists(motion_TTT_results_path):
                                        run_alt_opt(args)





