
import os
import logging
import pickle
import traceback
import numpy as np

from functions.utils.helpers.helpers_getargs import get_args

from functions.utils.helpers.helpers_init import initialize_directories, init_alt_opt, init_L1min

from functions.baselines_src.alt_opt_simData_module import AltOptModuleSim
from functions.baselines_src.L1min_simData_module import L1minModuleSim


def run_alt_opt(args):

    init_alt_opt(args)

    logging.info(f"\nRun experiment {args.experiment_name_alt_opt} with seed {args.seed} on gpu {args.gpu}.\n")
    logging.info(f"Apply alternating optimization to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    alt_opt_module = AltOptModuleSim(args)
    logging.info("Starting alternating optimization recon")

    alt_opt_module.run_alt_opt()

def run_L1min(args):

    init_L1min(args)
    logging.info(f"Start L1-minimization with args: {args}")

    L1min_module = L1minModuleSim(args)
    logging.info(f"Apply to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    logging.info(f"\nRun experiment {args.experiment_name_L1min} in run folder {args.experiment_run_folder_name} with seed {args.seed} on gpu {args.gpu}.\n")

    L1min_module.run_L1min()

if __name__ == '__main__':

    args = get_args()

    data_drive= "/media/ssd1/"

    path_to_data = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel"
    results_dir = "alt_opt_recons"
    path_to_result_dir = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/motion_MRI_TTT_results_tobit_kun"

    #dataset_path = "data_files/volume_dataset_freqEnc170_test_len10.pickle"
    dataset_path = "data_files/volume_dataset_freqEnc170_val_len4.pickle"

    with open(dataset_path,'rb') as fn:
        examples_list_of_dicts = pickle.load(fn)

    for Ns in [50]:
        for num_motion_events in [5]:
            for example_dict in examples_list_of_dicts[0:1]:
                filename = example_dict["filename"]   
                for max_mot in [5]:
                    for random_motion_seed_ind in ['seed1']: # 'seed1', 'seed2', 'seed3', 'seed4', 'seed5'
                        random_motion_seed = example_dict['motion_seeds'][random_motion_seed_ind]
                        
                        # Set data paths
                        args.example_path = f"{path_to_data}/Val_converted/{filename}"
                        args.mask_path= "data_files/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
                        args.sensmaps_path = f"{path_to_data}/Val_s_maps_3D/smaps_{filename}"
                        
                        # Set device index
                        args.gpu= 3

                        # Run alternating optimization
                        args.alt_opt=True
                        args.alt_opt_vivo=False #alt_opt also needs to be True

                        args.seed = 1

                        # Give an additional name for a folder that then contains a set of experiments
                        args.experiment_run_folder_name = "val_playground/"

                        args.nufft_norm = None

                        # # alternating optimization arguments
                        args.altopt_recon_only=False
                        args.altopt_motion_estimation_only = False
                        args.altopt_steps_total = 500
                        args.altopt_nufft_max_coil_size = 6
                        
                        args.altopt_steps_recon = 2
                        args.altopt_lr_recon = 5e7 
                        args.altopt_lam_recon = 1e-3 
                        args.altopt_optimizer_recon = 'SGD'

                        args.altopt_steps_motion = 4
                        args.altopt_lr_motion = 5e-11
                        args.altopt_optimizer_motion = 'SGD'

                        # # Motion and sampling trajectory arguments
                        # Motion trajectory choices: 'uniform_interShot_event_model', 'uniform_intraShot_event_model'
                        args.Ns = Ns # must be equal to num_shots
                        args.center_in_first_state = True
                        args.fix_mot_maxksp_shot = False
                        args.motionTraj_simMot = "uniform_interShot_event_model" 
                        args.num_shots = Ns 
                        args.num_motion_events = num_motion_events
                        args.num_intraShot_events = int(np.ceil(num_motion_events/2))
                        args.max_trans = max_mot
                        args.max_rot = max_mot
                        args.random_motion_seed = random_motion_seed

                        # Sampling trajectory choices: 
                        # 'random_cartesian', 'deterministic_cartesian', 'interleaved_cartesian', 'linear_cartesian
                        args.sampTraj_simMot = "interleaved_cartesian"
                        args.random_sampTraj_seed = 0 # for random_cartesian
                        args.sampling_order_path = "data_files/mask_3D_size_218_170_256_R_4_poisson_disc_traj_DS5_MODEquadruples_center3_order03142.pickle"

                        args.experiment_name_alt_opt= f"_rottrans{max_mot}_Ns{Ns}_motEvents{num_motion_events}_motSeed{random_motion_seed}"

                        # initialize experiment, create directories
                        args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")

                        if args.alt_opt:
                            run_alt_opt(args)

                        # Run L1-minimization after alt opt based on estimated motion parameters
                        # with/without DC thresholding and aligning motion parameter.                          
                        args.L1min=True
                        args.L1min_vivo=False 
                        args.L1min_mode = 'pred_mot_altopt'

                        args.L1min_optimizer = 'SGD'
                        args.L1min_lr = 5e7
                        args.L1min_lambda = 1e-3
                        args.L1min_num_steps = 50
                        args.L1min_DC_threshold = 0.575

                        # Control gpu memory consumption
                        args.L1min_nufft_max_coil_size = 12    
                        
                        if args.L1min:
                            for L1min_DC_loss_thresholding in [True, False]:
                                for L1min_motion_alignment in [True, False]:
                                                
                                    args.Ns = Ns
                                    args.L1min_DC_loss_thresholding = L1min_DC_loss_thresholding
                                    args.L1min_motion_alignment = L1min_motion_alignment
                                    args.experiment_name_L1min= f"_dcTh{args.L1min_DC_threshold}_{args.L1min_DC_loss_thresholding}_align_{args.L1min_motion_alignment}"
                                    args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")
                                    run_L1min(args)


