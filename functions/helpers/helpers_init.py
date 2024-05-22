import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter



def initialize_directories(args):

    args.experiment_path = os.path.join(args.data_drive,args.save_exp_results_path, args.experiment_name)

    if args.TTT or args.train:
        if os.path.exists(args.experiment_path) and args.train and args.load_model_from_checkpoint == "None":
            raise Exception("Experiment path already exists, training is set true and no checkpoint for loading specified.")

        os.makedirs(args.experiment_path, exist_ok=True)

        args.checkpoint_path = os.path.join(args.experiment_path, "checkpoints")
        os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.train:
        args.train_results_path = os.path.join(args.experiment_path, "train_results")
        os.makedirs(args.train_results_path, exist_ok=True)
        args.train_log_path = os.path.join(args.train_results_path, "train_log.log")

    if args.TTT or args.alt_opt_on_TTTexp:
        filename = args.TTT_example_path.split("/")[-1].split(".")[0]
        if args.finalTestset:
            args.TTT_results_path = os.path.join(args.experiment_path+"/finalTestset", f"TTT_results_{filename}"+args.experiment_name_TTT)
        else:
            args.TTT_results_path = os.path.join(args.experiment_path, f"{args.experiment_run_folder_name}TTT_results_{filename}"+args.experiment_name_TTT)
        
        os.makedirs(args.TTT_results_path, exist_ok=True)
        args.TTT_log_path = os.path.join(args.TTT_results_path, "TTT_log.log")

        args.TTT_results_path_numerical = os.path.join(args.TTT_results_path, "numerical")
        os.makedirs(args.TTT_results_path_numerical, exist_ok=True)


    if (args.alt_opt and not args.alt_opt_on_TTTexp) or args.alt_opt_on_alt_opt_exp:
        filename = args.TTT_example_path.split("/")[-1].split(".")[0]
        if args.finalTestset:
            args.altopt_results_path = os.path.join(args.data_drive,args.save_exp_results_path+"finalTestset/", f"{filename}"+args.experiment_name_alt_opt)
        else:
            args.altopt_results_path = os.path.join(args.data_drive,args.save_exp_results_path, f"{args.experiment_run_folder_name}{filename}"+args.experiment_name_alt_opt)
        
        if args.alt_opt_on_alt_opt_exp:
            assert args.altopt_recon_only == True
            args.altopt_load_path = args.altopt_results_path
            args.altopt_results_path = os.path.join(args.altopt_results_path, args.experiment_name_alt_opt_on_altopt)

        os.makedirs(args.altopt_results_path, exist_ok=True)
        args.altopt_log_path = os.path.join(args.altopt_results_path, "alt_opt_log.log")

    if args.alt_opt_on_TTTexp:
        filename = args.TTT_example_path.split("/")[-1].split(".")[0]

        if args.finalTestset:
            args.altopt_results_path = os.path.join(args.experiment_path+"/finalTestset", f"TTT_results_{filename}"+args.experiment_name_TTT+f"/_l1_recon/{args.experiment_name_alt_opt}")
        else:
            args.altopt_results_path = os.path.join(args.experiment_path, f"{args.experiment_run_folder_name}TTT_results_{filename}"+args.experiment_name_TTT+f"/_l1_recon/{args.experiment_name_alt_opt}")
        
        os.makedirs(args.altopt_results_path, exist_ok=True)
        args.altopt_log_path = os.path.join(args.altopt_results_path, "alt_opt_log.log")


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
            logging.info(f"Using multistep scheduler with milestones {args.multistep_milestones} and gamma {args.multistep_gamma}")
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=args.num_epochs, 
                eta_min=args.cosine_min_lr, 
                last_epoch=-1, 
                verbose=False
                )
            logging.info(f"Using cosine scheduler with min lr {args.cosine_min_lr}")

        else:
            scheduler = None
    else:
        scheduler = None


    return optimizer, scheduler

        