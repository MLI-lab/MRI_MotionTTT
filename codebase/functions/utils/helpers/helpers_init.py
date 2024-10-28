import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


def initialize_directories(args, results_path):

    

    if args.train:
        args.experiment_path_train = os.path.join(results_path, args.experiment_name_train)
        #if os.path.exists(args.experiment_path_train) and args.train and args.load_model_from_checkpoint == "None" and args.load_model_path == "None":
        #    raise Exception("Experiment path already exists, training is set true and no checkpoint for loading specified.")

        os.makedirs(args.experiment_path_train, exist_ok=True)

        args.checkpoint_path = os.path.join(args.experiment_path_train, "checkpoints")
        os.makedirs(args.checkpoint_path, exist_ok=True)

        args.train_results_path = os.path.join(args.experiment_path_train, "train_results")
        os.makedirs(args.train_results_path, exist_ok=True)
        args.train_log_path = os.path.join(args.train_results_path, "train_log.log")

    if args.eval_valset:
        args.train_results_path = os.path.join(results_path)
        os.makedirs(args.train_results_path, exist_ok=True)
        args.train_log_path = os.path.join(args.train_results_path, "evaluate_valset_log.log")

    if args.TTT:
        filename = args.example_path.split("/")[-1].split(".")[0]

        args.TTT_results_path = os.path.join( results_path, args.experiment_run_folder_name, f"TTT_results_{filename}"+args.experiment_name_TTT)
        
        os.makedirs(args.TTT_results_path, exist_ok=True)
        args.TTT_log_path = os.path.join(args.TTT_results_path, "TTT_log.log")

        args.TTT_results_path_numerical = os.path.join(args.TTT_results_path, "numerical")
        os.makedirs(args.TTT_results_path_numerical, exist_ok=True)

    if args.alt_opt:
        filename = args.example_path.split("/")[-1].split(".")[0]

        args.altopt_results_path = os.path.join( results_path, args.experiment_run_folder_name, filename+args.experiment_name_alt_opt)
        
        os.makedirs(args.altopt_results_path, exist_ok=True)
        args.altopt_log_path = os.path.join(args.altopt_results_path, "alt_opt_log.log")

    if args.L1min:
        filename = args.example_path.split("/")[-1].split(".")[0]

        if args.L1min_mode == 'pred_mot_motionTTT':

            args.L1min_load_path = os.path.join(results_path, args.experiment_run_folder_name,f"TTT_results_{filename}"+args.experiment_name_TTT, "numerical")
            args.L1min_load_path = os.path.join(args.L1min_load_path, f"phase{args.L1min_on_TTT_load_from_phase}/final_result_dict.pkl")
            
            args.L1min_results_path = os.path.join(results_path, args.experiment_run_folder_name,f"TTT_results_{filename}"+args.experiment_name_TTT, "l1_recon", args.experiment_name_L1min)

        elif args.L1min_mode == 'gt_motion' or args.L1min_mode == 'noMoCo':
            args.L1min_results_path = os.path.join(results_path, args.experiment_run_folder_name, f"{filename}{args.experiment_name_L1min}")

        elif args.L1min_mode == 'pred_mot_altopt':
            args.L1min_load_path = os.path.join(results_path, args.experiment_run_folder_name, f"{filename}{args.experiment_name_alt_opt}", "final_result_dict.pkl")

            args.L1min_results_path = os.path.join(results_path, args.experiment_run_folder_name, f"{filename}{args.experiment_name_alt_opt}", "l1_recon", args.experiment_name_L1min)

        else:
            raise Exception("L1min mode not implemented.")
        
        os.makedirs(args.L1min_results_path, exist_ok=True)
        args.L1min_log_path = os.path.join(args.L1min_results_path, "L1min_log.log")

    return args


def init_training(args):

    if args.log_to_tb:
        tb_writer = SummaryWriter(args.train_results_path)
    else:
        tb_writer = None

    init_logging(args.train_log_path)
    
    return tb_writer

def init_TTT(args):

    init_logging(args.TTT_log_path)

def init_alt_opt(args):

    init_logging(args.altopt_log_path)

def init_L1min(args):

    init_logging(args.L1min_log_path)


def init_logging(path_to_log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]

    mode = "a" if os.path.exists(path_to_log_file) else "w"
    handlers.append(logging.FileHandler(path_to_log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def init_optimization(args, model, mode):
    if mode == "train": 
        lr = args.lr
    elif mode == "TTT":
        lr = args.lr_TTT

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        logging.info(f"Using Adam optimizer with learning rate {lr}")
    else:
        raise Exception("Optimizer not implemented.")

    if mode == "train":
        if args.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=args.multistep_milestones, 
                gamma=args.multistep_gamma, 
                last_epoch=- 1, 
                verbose=False
                )
            logging.info(f"Using multistep scheduler with milestones {args.multistep_milestones}, gamma {args.multistep_gamma} and train for {args.num_epochs} epochs")
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=args.num_epochs, 
                eta_min=args.cosine_min_lr, 
                last_epoch=-1, 
                verbose=False
                )
            logging.info(f"Using cosine scheduler with min lr {args.cosine_min_lr} and train for {args.num_epochs} epochs")

        else:
            scheduler = None
    else:
        scheduler = None


    return optimizer, scheduler

        