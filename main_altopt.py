
import os
import logging
import torch
import pickle
import traceback

from functions.helpers.helpers_getargs import get_args

from functions.helpers.helpers_init import initialize_directories, init_alt_opt

from functions.test_time_training.alt_opt_module import AltOptModule
from functions.test_time_training.alt_opt_vivo import AltOptModuleVivo


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

    args.data_drive= "/media/ssd1/"

    #args.finalTestset_data_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/volume_dataset_freqEnc170_test_len5_1.pickle"
    args.valset_data_path = "volume_dataset_freqEnc170_val_len4.pickle"
    with open(os.path.join(args.data_drive, args.valset_data_path),'rb') as fn:
        examples_list_of_dicts = pickle.load(fn)

    for Ns in [50]:
        for TTT_num_shots in [50]:
            for TTT_states_per_split in [5]:
                for num_motion_events in [5]:
                    for example_dict in examples_list_of_dicts[0:1]:
                        filename = example_dict["filename"]
                        
                        for max_mot in [5]:
                            for random_motion_seed_ind in ['seed1']:
                                random_motion_seed = example_dict['motion_seeds'][random_motion_seed_ind]
                                
                                args.TTT_example_path = f"cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/{filename}"
                                

                                args.TTT_mask_path= "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
                                
                                args.gpu= 2
                                args.alt_opt=True
                                args.alt_opt_vivo=False #alt_opt also needs to be True

                                args.finalTestset = False

                                args.seed = 1

                                # Give an additional name for a folder that then contains a set of experiments
                                args.experiment_run_folder_name = "val_playground/"

                                args.nufft_norm = None

                                # If TTT_intraShot_estimation_only is True then in the first iteration the predicted motion
                                # parameters are set to ground truth values except for shots at which intra-shot motion is occurs.
                                # For those shots the predicted motion parameters are split according to TTT_states_per_split and initilized with zeros.
                                # ALSO can be set for alternating optimization !
                                args.TTT_intraShot_estimation_only = False
                                
                                # # alternating optimzation arguments on top of a alternating optimization experiment
                                args.alt_opt_on_alt_opt_exp = False
                                args.experiment_name_alt_opt_on_altopt = "reconOnly_lrRec5e7_lamRec1e-3_SGD50_BSnufft6_alignTrue"

                                # # alternating optimization arguments
                                args.altopt_recon_only=True
                                args.altopt_dc_thresholding = False
                                args.altopt_dc_threshold = 0.62
                                args.altopt_align_motParams = False # only important for simulated motion
                                args.altopt_motion_estimation_only=False
                                args.altopt_recon_only_with_motionKnowledge = True
                                args.altopt_steps_total = 50#100
                                args.altopt_nufft_max_coil_size = 6
                                
                                args.altopt_steps_recon = 1#2
                                args.altopt_lr_recon = 5e7 # 5e7 for train data
                                args.altopt_lam_recon = 1e-3 #1e-3 for train data
                                args.altopt_optimizer_recon = 'SGD'

                                args.altopt_steps_motion = 0#4
                                args.altopt_lr_motion = 5e-11
                                args.altopt_optimizer_motion = 'SGD'

                                # # Motion and sampling trajectory arguments
                                # Motion trajectory choices: 'uniform_interShot_event_model', 'uniform_intraShot_event_model'
                                args.Ns = Ns # currently must be equal to TTT_num_shots
                                args.center_in_first_state = True
                                args.fix_mot_maxksp_shot = False
                                args.motionTraj_simMot = "uniform_interShot_event_model" 
                                args.TTT_num_shots = TTT_num_shots 
                                args.num_motion_events = num_motion_events
                                args.num_intraShot_events = 5
                                args.max_trans = max_mot
                                args.max_rot = max_mot
                                args.random_motion_seed = random_motion_seed

                                # Sampling trajectory choices: 
                                # 'interleaved_cartesian_Ns500', 'random_cartesian', 'deterministic_cartesian', 'interleaved_cartesian', 'linear_cartesian
                                args.TTT_sampTraj_simMot = "deterministic_cartesian"
                                args.random_sampTraj_seed = 0 # for random_cartesian
                                args.sampling_order_path = "cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/mask_3D_size_218_170_256_R_4_poisson_disc_traj_DS5_MODEquadruples_center3_order03142.pickle"

                                args.experiment_name_alt_opt= f"_rottrans{max_mot}_Ns{Ns}_Nshots{TTT_num_shots}_motEvents{num_motion_events}_intraEvents{args.num_intraShot_events}_motSeed{random_motion_seed}_reconOnly_oldSetup"

                                # initialize experiment, create directories
                                args = initialize_directories(args)

                                if args.alt_opt:
                                    run_alt_opt(args)


                                    if not args.altopt_recon_only and not args.altopt_motion_estimation_only:
                                        args.alt_opt_on_alt_opt_exp = True
                                        args.altopt_recon_only=True
                                        args.altopt_steps_total = 2#50
                                        args.altopt_steps_recon = 1
                                        args.altopt_lr_recon = 5e7 # 5e7 for train data, 1e8 for scan data
                                        args.altopt_lam_recon = 1e-3 #1e-3 for train data, 3e-8 for scan data
                                        args.altopt_optimizer_recon = 'SGD'

                                        args.altopt_steps_motion = 0

                                        for altopt_dc_thresholding in [True, False]:
                                            for altopt_align_motParams in [True, False]:
                                                for phase in [0]:
                                                    try:                                                
                                                        args.Ns = Ns
                                                        #args.alt_opt_on_TTT_load_from_phase = phase
                                                        args.altopt_dc_thresholding = altopt_dc_thresholding
                                                        args.altopt_align_motParams = altopt_align_motParams
                                                        args.experiment_name_alt_opt_on_altopt= f"phase{phase}_dcTh{args.altopt_dc_threshold}_{args.altopt_dc_thresholding}_align_{args.altopt_align_motParams}"
                                                        args = initialize_directories(args)
                                                        run_alt_opt(args)
                                                    except:
                                                        logging.info(f"Phase {phase} failed due to:")
                                                        error_str = traceback.format_exc()
                                                        logging.info(error_str)  
                                                        pass







