
import os
import logging
import traceback

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

import sys

# setting path
#sys.path.append('../')

from functions.utils.helpers.helpers_getargs import get_args

from functions.utils.helpers.helpers_init import initialize_directories, init_TTT, init_L1min
from functions.utils.models.helpers_model import get_model

from functions.motionTTT_src.unet_TTT_vivoData_module import UnetTTTModuleVivo
from functions.baselines_src.L1min_vivoData_module import L1minModuleVivo

def run_TTT(args):

    init_TTT(args)
    logging.info(f"Start motionTTT with args: {args}")

    logging.info(f"\nRun with seed {args.seed} on gpu {args.gpu}.\n")

    with open(os.path.join(args.TTT_results_path, "args.txt"), "w") as f:
        f.write(str(args))

    logging.info(f"Apply TTT to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    model, _, _, _ = get_model(args)

    if args.model == "unet":
        TTT_module = UnetTTTModuleVivo(args, model)
        logging.info("Starting simulated motion TTT")

    TTT_module.run_TTT()

def run_L1min(args):

    init_L1min(args)
    logging.info(f"Start L1-minimization with args: {args}")

    L1min_module = L1minModuleVivo(args)
    logging.info(f"Apply to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    logging.info(f"\nRun experiment {args.experiment_name_L1min} in run folder {args.experiment_run_folder_name} with seed {args.seed} on gpu {args.gpu}.\n")

    L1min_module.run_L1min()


if __name__ == '__main__':
    args = get_args()

    data_drive= "/media/ssd0/"

    path_to_data = "/media/ssd0/PMoC3D/dervatives/cropped_data"
    path_to_result_dir = "/media/ssd0/PMoC3D/dervatives/recon_folder"
    results_dir = "MotionTTT_PMoC3D"

    for sub_ind in range(1,9):
        for scan_ind in range(4):
            # Set data paths
            args.example_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_run-0{scan_ind}_kspace.h5')
            args.sensmaps_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_smaps.h5')
            # Give an additional name for a folder that then contains a set of experiments
            args.experiment_run_folder_name = f"sub-0{sub_ind}/"

            # Set device index
            args.gpu= 3

            # Run motionTTT
            args.TTT=True
            args.TTT_vivo=True # TTT also needs to be True

            args.seed = 1

            # Nufft arguments
            args.nufft_norm = None
            args.TTT_use_nufft_with_dcomp = True # use adjoint nufft with density compensation

            # # Model arguments to load a pre-trained model
            # Load a local model:
            args.load_model_path= 'None'
            args.model= 'unet'
            args.chans= 48
            # Load a model from huggingface:
            args.load_model_from_huggingface = "mli-lab/Unet48-2D-CC359"

            # # motion TTT arguments
            
            # Optimization arguments
            args.lr_TTT= 1.0 
            args.TTT_optimizer = 'Adam'
            args.TTT_lr_max_decays = 4
            args.TTT_lr_decay_factor = 0.5
            args.TTT_lr_decay_at_the_latest = [45,55,65,1000]
            args.num_steps_TTT= 75
            args.TTT_all_axes= True
            args.TTT_num_rot_only_grad_steps = 5
            args.TTT_use_clamp_schedule = True

            # Control GPU memory consumption and computational time
            args.num_slices_per_grad_step= 5
            args.TTT_motState_batchSize_per_backprop = None # None for full batch
            args.TTT_nufft_max_coil_size = 7 # None for full batch

            # Phase 2: Specify iterations at which states detected by TTT_DCloss_th_split are split
            # into TTT_states_per_split states. If TTT_all_states_grad_after_split is False
            # then only the states that are split are optimized over for the remainin iterations
            args.TTT_list_of_split_steps = []
            args.TTT_states_per_split = 10
            args.TTT_all_states_grad_after_split = False
            args.TTT_lr_after_split = 0.5
            args.TTT_DCloss_th_split = 0.70

            # Phase 3: Set an interation after which motionTTT defines a new optimizer and starts optimizing
            # over all motion states in the current pred_motion_params for the remaining iterations
            args.TTT_optimize_all_states_after = 100
            args.TTT_optimize_all_states_after_lr = 0.05

            # # Motion arguments
            args.Ns = 52 # at init number of motion states Ns must be equal to num_shots
            args.fix_mot_maxksp_shot = False

            args.experiment_name_TTT= f"_p1_lr1.0_45_55_65_75_dcTh0.70"

            # initialize experiment, create directories
            args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")

            # Run motionTTT
            if args.TTT:
                run_TTT(args)

            # Run L1-minimization after motionTTT based on estimated motion parameters
            # with/without DC thresholding and aligning motion parameters for estimated
            # motion states from each phase of motionTTT.                          
            args.L1min=True
            args.L1min_vivo=True 
            args.L1min_mode = 'pred_mot_motionTTT'

            args.L1min_optimizer = 'SGD'
            args.L1min_lr = 1e8
            args.L1min_lambda = 3e-8
            args.L1min_num_steps = 50
            args.L1min_DC_threshold = 0.70

            # Control gpu memory consumption
            args.L1min_nufft_max_coil_size = 7
            
            if args.L1min:
                for L1min_DC_loss_thresholding in [True, False]:
                        for phase in [0]:
                            try:                                                
                                args.Ns = 52
                                args.L1min_on_TTT_load_from_phase = phase
                                args.L1min_DC_loss_thresholding = L1min_DC_loss_thresholding
                                args.experiment_name_L1min= f"phase{args.L1min_on_TTT_load_from_phase}_dcTh{args.L1min_DC_threshold}_{args.L1min_DC_loss_thresholding}"
                                args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")
                                run_L1min(args)
                            except:
                                logging.info(f"Phase {phase} failed due to:")
                                error_str = traceback.format_exc()
                                logging.info(error_str)  
                                pass




