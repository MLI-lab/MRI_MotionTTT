
import os
import logging

from functions.utils.helpers.helpers_getargs import get_args

from functions.utils.helpers.helpers_init import initialize_directories, init_alt_opt, init_L1min

from functions.baselines_src.alt_opt_vivoData_module import AltOptModuleVivo
from functions.baselines_src.L1min_vivoData_module import L1minModuleVivo


def run_alt_opt(args):

    init_alt_opt(args)

    logging.info(f"\nRun experiment {args.experiment_name_alt_opt} with seed {args.seed} on gpu {args.gpu}.\n")
    logging.info(f"Apply alternating optimization to example from {args.example_path}")
    logging.info(f"Load sensitivity maps from {args.sensmaps_path}")

    alt_opt_module = AltOptModuleVivo(args)
    logging.info("Starting alternating optimization recon")

    alt_opt_module.run_alt_opt()

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
    results_dir = "alt_opt_recons"

    for sub_ind in range(1,9):
        for scan_ind in range(4):
            # Set data paths
            args.example_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_run-0{scan_ind}_kspace.h5')
            args.sensmaps_path = os.path.join(path_to_data,f'sub-0{sub_ind}',f'sub-0{sub_ind}_smaps.h5')
            # Give an additional name for a folder that then contains a set of experiments
            args.experiment_run_folder_name = f"sub-0{sub_ind}/"
                            
            # Set device index
            args.gpu= 3

            # Run alternating optimization
            args.alt_opt=True
            args.alt_opt_vivo=True #alt_opt also needs to be True

            args.seed = 1

            args.nufft_norm = None

            # # alternating optimization arguments
            args.altopt_steps_total = 500
            args.altopt_nufft_max_coil_size = 7
            
            args.altopt_steps_recon = 2
            args.altopt_lr_recon = 1e8 
            args.altopt_lam_recon = 3e-8 
            args.altopt_optimizer_recon = 'SGD'

            args.altopt_steps_motion = 4
            args.altopt_lr_motion = 5e-2
            args.altopt_optimizer_motion = 'SGD'

            # # Motion arguments
            args.Ns = 52 # at init number of motion states Ns must be equal to num_shots
            args.fix_mot_maxksp_shot = False

            # Args for the early stopping:
            args.is_altopt_threshold = True
            args.altopt_threshold = 0.02 #13
            args.alt_opt_threshold_mode = 'end_of_recon_step' # 'end_of_recon_step' (for vivo mot) or 'end_of_iteration' (for sim mot)

            args.experiment_name_alt_opt= f"_rec_lr{args.altopt_lr_recon:.1e}_lam{args.altopt_lam_recon:.1e}_s{args.altopt_steps_recon}_mot_lr{args.altopt_lr_motion:.1e}_s{args.altopt_steps_motion}_es{args.is_altopt_threshold}_th{args.altopt_threshold}"

            # initialize experiment, create directories
            args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")

            if args.alt_opt:
                run_alt_opt(args)

            # Run L1-minimization after alt opt based on estimated motion parameters
            # with/without DC thresholding and aligning motion parameter.                          
            args.L1min=True
            args.L1min_vivo=True 
            args.L1min_mode = 'pred_mot_altopt'

            args.L1min_optimizer = 'SGD'
            args.L1min_lr = 1e8
            args.L1min_lambda = 3e-8
            args.L1min_num_steps = 50
            args.L1min_DC_threshold = 0.65

            # Control gpu memory consumption
            args.L1min_nufft_max_coil_size = 2
            
            if args.L1min:
                for L1min_DC_loss_thresholding in [True, False]:
                    args.Ns = 52
                    args.L1min_DC_loss_thresholding = L1min_DC_loss_thresholding
                    args.experiment_name_L1min= f"dcTh{args.L1min_DC_threshold}_{args.L1min_DC_loss_thresholding}"
                    args = initialize_directories(args, results_path = path_to_result_dir+"/"+results_dir+"/")
                    run_L1min(args)

