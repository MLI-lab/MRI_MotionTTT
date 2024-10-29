
import os
import logging
import pickle
import traceback
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

import sys

# setting path
#sys.path.append('../')

from functions.utils.helpers.helpers_getargs import get_args

from functions.utils.helpers.helpers_init import initialize_directories, init_TTT, init_L1min
from functions.utils.models.helpers_model import get_model

from functions.motionTTT_src.unet_TTT_simData_module import UnetTTTModuleSim
from functions.baselines_src.L1min_simData_module import L1minModuleSim

def run_TTT(args):

    init_TTT(args)
    logging.info(f"Start motionTTT with args: {args}")

    logging.info(f"\nRun with seed {args.seed} on gpu {args.gpu}.\n")
    logging.info(f"Nufft settings are: nufft norm {args.nufft_norm}, use nufft with dcomp {args.TTT_use_nufft_with_dcomp}, and nufft max coil size {args.TTT_nufft_max_coil_size}.")

    with open(os.path.join(args.TTT_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    logging.info(f"Apply TTT to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    model, _, _, _ = get_model(args)

    if args.model == "unet":
        TTT_module = UnetTTTModuleSim(args, model)
        logging.info("Starting simulated motion TTT")

    TTT_module.run_TTT()

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

    data_drive= "/media/ssd0/"

    path_to_data = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel"
    results_dir = "E015_unet48_PmaskR4_lr001_joint_nufftAdjoint_with_dcomp"
    path_to_result_dir = f"{data_drive}cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/motion_MRI_TTT_results_tobit_kun"

    #dataset_path = "data_files/volume_dataset_freqEnc170_test_len10.pickle"
    dataset_path = "data_files/volume_dataset_freqEnc170_val_len4.pickle"

    with open(dataset_path,'rb') as fn:
        examples_list_of_dicts = pickle.load(fn)

    for Ns in [50]:
        for TTT_states_per_split in [1]:
            for num_motion_events in [5]:
                for example_dict in examples_list_of_dicts[0:1]:
                    filename = example_dict["filename"]
                    for max_mot in [10]:
                        for random_motion_seed_ind in ['seed1']: # 'seed1', 'seed2', 'seed3', 'seed4', 'seed5'
                            random_motion_seed = example_dict['motion_seeds'][random_motion_seed_ind]
                            
                            # Set data paths
                            args.example_path = f"{path_to_data}/Val_converted/{filename}"
                            args.mask_path= "data_files/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
                            args.sensmaps_path = f"{path_to_data}/Val_s_maps_3D/smaps_{filename}"
                            
                            # Set device index
                            args.gpu= 0

                            # Run motionTTT
                            args.TTT=True
                            args.TTT_vivo=False # TTT also needs to be True

                            args.seed = 1

                            # Give an additional name for a folder that then contains a set of experiments
                            args.experiment_run_folder_name = "val_playground_4/"

                            # # Model arguments to load a pre-trained model
                            # Load a local model:
                            args.load_model_path= 'None'
                            args.model= 'unet'
                            args.chans= 48
                            # Load a model from huggingface:
                            args.load_model_from_huggingface = "mli-lab/Unet48-2D-CC359"

                            # # motion TTT arguments
                            
                            # Optimization arguments
                            args.lr_TTT= 4.0 
                            args.TTT_optimizer = 'Adam'
                            args.TTT_lr_max_decays = 3
                            args.TTT_lr_decay_factor = 0.25
                            args.TTT_lr_decay_at_the_latest = [39,59,1000]
                            args.num_steps_TTT= 130
                            args.TTT_all_axes= True
                            args.TTT_num_rot_only_grad_steps = 5
                            args.TTT_use_clamp_schedule = True
                            args.num_slices_per_grad_step= 5

                            # Control GPU memory consumption and computational time
                            args.TTT_motState_batchSize_per_backprop = 25 # None for full batch
                            args.TTT_nufft_max_coil_size = 4 # None for full batch

                            # Phase 2: Specify iterations at which states detected by TTT_DCloss_th_split are split
                            # into TTT_states_per_split states. If TTT_all_states_grad_after_split is False
                            # then only the states that are split are optimized over for the remainin iterations
                            args.TTT_list_of_split_steps = [71]
                            args.TTT_states_per_split = TTT_states_per_split
                            args.TTT_all_states_grad_after_split = False
                            args.TTT_lr_after_split = 0.5
                            args.TTT_DCloss_th_split = 0.575

                            # Phase 3: Set an interation after which motionTTT defines a new optimizer and starts optimizing
                            # over all motion states in the current pred_motion_params for the remaining iterations
                            args.TTT_optimize_all_states_after = 100
                            args.TTT_optimize_all_states_after_lr = 0.05


                            # # Motion and sampling trajectory arguments
                            # Motion trajectory choices: 'uniform_interShot_event_model', 'uniform_intraShot_event_model'
                            args.Ns = Ns # at init number of motion states Ns must be equal to num_shots
                            args.center_in_first_state = True
                            args.fix_mot_maxksp_shot = True
                            args.motionTraj_simMot = "uniform_interShot_event_model" 
                            args.num_shots = Ns 
                            args.num_motion_events = num_motion_events
                            args.num_intraShot_events = int(np.ceil(num_motion_events/2))
                            args.max_trans = max_mot
                            args.max_rot = max_mot
                            args.random_motion_seed = random_motion_seed

                            # Sampling trajectory choices: 
                            # 'random_cartesian', 'interleaved_cartesian', 'deterministic_cartesian', 'linear_cartesian'
                            args.sampTraj_simMot = "interleaved_cartesian"
                            args.random_sampTraj_seed = 0 # for random_cartesian
                            args.sampling_order_path = "data_files/mask_3D_size_218_170_256_R_4_poisson_disc_traj_DS5_MODEquadruples_center3_order03142.pickle" # for deterministic_cartesian

                            # For intra-shot
                            #args.experiment_name_TTT= f"_rottrans{max_mot}_Ns{Ns}_motEvents{num_motion_events}_intraEvents{args.num_intraShot_events}_motSeed{random_motion_seed}_splits{args.TTT_states_per_split}"
                            # For inter-shot
                            args.experiment_name_TTT= f"_rottrans{max_mot}_Ns{Ns}_motEvents{num_motion_events}_motSeed{random_motion_seed}_splits{args.TTT_states_per_split}"

                            # initialize experiment, create directories
                            args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")

                            # Run motionTTT
                            if args.TTT:
                                run_TTT(args)

                            # Run L1-minimization after motionTTT based on estimated motion parameters
                            # with/without DC thresholding and aligning motion parameters for estimated
                            # motion states from each phase of motionTTT.                          
                            args.L1min=True
                            args.L1min_vivo=False 
                            args.L1min_mode = 'pred_mot_motionTTT'

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
                                        for phase in [0,1,2]:
                                            try:                                                
                                                args.Ns = Ns
                                                args.L1min_on_TTT_load_from_phase = phase
                                                args.L1min_DC_loss_thresholding = L1min_DC_loss_thresholding
                                                args.L1min_motion_alignment = L1min_motion_alignment
                                                args.experiment_name_L1min= f"phase{args.L1min_on_TTT_load_from_phase}_dcTh{args.L1min_DC_threshold}_{args.L1min_DC_loss_thresholding}_align_{args.L1min_motion_alignment}"
                                                args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")
                                                run_L1min(args)
                                            except:
                                                logging.info(f"Phase {phase} failed due to:")
                                                error_str = traceback.format_exc()
                                                logging.info(error_str)  
                                                pass





