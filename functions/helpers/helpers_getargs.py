from argparse import ArgumentParser
import sys
import torch

def get_args():

    parser = ArgumentParser()

    parser.add_argument(
        '--experiment_name',
        default="E999_test",
        type=str,
        help='Name of experiment. Specifies the model checkpoint to load for TTT.'
    )

    parser.add_argument(
        '--experiment_run_folder_name',
        default="",
        type=str,
        help='Name of folder to save experiment results. Must end with /'
    )

    parser.add_argument(
        '--experiment_name_TTT',
        default="",
        type=str,
        help='Name of TTT experiment. E.g., _onlyTransl'
    )

    parser.add_argument(
        '--experiment_name_alt_opt',
        default="",
        type=str,
        help='Name of alt opt experiment. E.g., _lr001'
    )

    parser.add_argument(
        '--num_test_examples',
        default=None,
        type=int,
        help='Number of test and TTT examples to use. If None, use all.'
    )

    parser.add_argument(
        '--test_run',
        default=False,
        action='store_true',
        help='Test run: 2 examples per dataset, 2 epochs.'
    )

    parser.add_argument(
        '--finalTestset',
        default=False,
        action='store_true',
        help='Whether to evaluate on the final test set.'
    )    

    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='Whether to train the model.'
    )

    parser.add_argument(
        '--TTT',
        default=False,
        action='store_true',
        help='Whether to apply TTT to the model on the distribution-shift specific test set.'
    )

    parser.add_argument(
        '--alt_opt',
        default=False,
        action='store_true',
        help='Whether to apply alternating optimization.'
    )

    parser.add_argument(
        '--altopt_recon_only',
        default=False,
        action='store_true',
        help='Whether to apply only reconstruction within alternating optimization.'
    )

    parser.add_argument(
        '--altopt_motion_estimation_only',
        default=False,
        action='store_true',
        help='Whether to apply only motion estimation within alternating optimization.'
    )

    parser.add_argument(
        '--seed',
        default=1,
        type=int,
        help='Random seed for random, np.random, torch.cuda.manual_seed, torch.manual_seed.'
    )

    parser.add_argument(
        '--gpu',
        default=1,
        type=int,
        help='GPU to use.'
    )

    parser.add_argument(
        '--nufft_norm',
        default=None,
        type=str,
        choices=(None, 'ortho'),
        help='Normalization for NUFFT.'
    )

    parser = get_data_args(parser)
    parser = get_training_args(parser)
    parser = get_model_args(parser)
    parser = get_optimizer_args(parser)
    parser = get_TTT_args(parser)
    parser = get_args_motion(parser)
    parser = get_alt_opt_args(parser)

    args = parser.parse_args()

    return args

def secure_args_from_sys(args):

    # remove 'main.py'
    args_from_sys_raw = sys.argv[1:]

    # get keys
    keys = [arg[2:] for arg in args_from_sys_raw if arg[0:2] == "--"]
    args_from_sys = {}
    args_dict = vars(args)
    for key in keys:
        args_from_sys[key] = args_dict[key]

    return args_from_sys


def get_alt_opt_args(parser):

    parser.add_argument(
        '--altopt_steps_total',
        default=100,
        type=int,
        help='Number of recon steps per iteration.'
    )

    parser.add_argument(
        '--altopt_lr_recon',
        default=1e-4,
        type=float,
        help='Learning rate for recon problem.'
    )

    parser.add_argument(
        '--altopt_steps_recon',
        default=3,
        type=int,
        help='Number of recon steps per iteration.'
    )

    parser.add_argument(
        '--altopt_lam_recon',
        default=1e-3,
        type=float,
        help='Lambda for L1 regularization during recon.'
    )

    parser.add_argument(
        '--altopt_optimizer_recon',
        default='SGD',
        choices=('SGD', 'Adam'),
        type=str,
        help='Optimizer for recon problem.'
    )

    parser.add_argument(
        '--altopt_lr_motion',
        default=1e-4,
        type=float,
        help='Learning rate for motion est problem.'
    )

    parser.add_argument(
        '--altopt_steps_motion',
        default=3,
        type=int,
        help='Number of motion est steps per iteration.'
    )

    parser.add_argument(
        '--altopt_optimizer_motion',
        default='Adam',
        choices=('SGD', 'Adam'),
        type=str,
        help='Optimizer for motion est problem.'
    )

    parser.add_argument(
        '--altopt_recon_only_with_motionKnowledge',
        default=False,
        action='store_true',
        help='Whether to apply only reconstruction within alternating optimization with motion knowledge.'
    )

    parser.add_argument(
        '--alt_opt_on_TTTexp',
        default=False,
        action='store_true',
        help='Whether to load motion from TTT experiment and save results into TTT exp.'
    )

    parser.add_argument(
        '--altopt_nufft_max_coil_size',
        default=None,
        type=int,
        help='Maximum coil size for NUFFT during alt opt. None means all coils are processed in one batch.'
    )

    parser.add_argument(
        '--altopt_dc_thresholding',
        default=False,
        action='store_true',
        help='Whether to apply dc loss thresholding, i.e. ommit states with larger dc loss.'
    )

    parser.add_argument(
        '--altopt_dc_threshold',
        default=0.013,
        type=float,
        help='Threshold for dc loss.'
    )

    parser.add_argument(
        '--altopt_align_motParams',
        default=False,
        action='store_true',
        help='Whether to align motion parameters before reconstruction.'
    )

    parser.add_argument(
        '--alt_opt_on_alt_opt_exp',
        default=False,
        action='store_true',
        help='Whether to load motion from alt opt experiment and save results into alt opt exp.'
    )

    parser.add_argument(
        '--experiment_name_alt_opt_on_altopt',
        default="_",
        type=str,
        help='Name of alt opt experiment on alt opt experiment.'
    )

    return parser

def get_args_motion(parser):

    parser.add_argument(
        '--Ns',
        default=10,
        type=int,
        help='Number of motion states (shots). First state always has no motion.'
    )

    parser.add_argument(
        '--num_motion_events',
        default=10,
        type=int,
        help='Number of motion events, i.e. number of motion states with different motion parameters.'
    )

    parser.add_argument(
        '--max_trans',
        default=2,
        type=int,
        help='Translation between [-max_translation,max_translation] in pixel.'
    )

    parser.add_argument(
        '--max_rot',
        default=2,
        type=int,
        help='Rotation between [-max_rotation,max_rotation] in degrees.'
    )

    parser.add_argument(
        '--random_motion_seed',
        default=1,
        type=int,
        help='Random seed for random motion.'
    )

    return parser

def get_TTT_args(parser):

    parser.add_argument(
        '--TTT_from_which_checkpoint',
        default="best_SSIM",
        choices=("best_SSIM", "best_PSNR", "last"),
        type=str,
        help='Which checkpoint to load for TTT.'
    )

    parser.add_argument(
        '--lr_TTT',
        default=1e-5,
        type=float,
        help='Learning rate for TTT.'
    )

    parser.add_argument(
        '--num_steps_TTT',
        default=800,
        type=int,
        help='Number of fitting steps for TTT.'
    )

    parser.add_argument(
        '--num_slices_per_grad_step',
        default=3,
        type=int,
        help='Number of slices to use for each gradient step.'
    )

    parser.add_argument(
        '--TTT_all_axes',
        default=False,
        action='store_true',
        help='Whether use unet recon from all axes during TTT.'
    )


    parser.add_argument(
        '--TTT_num_rot_only_grad_steps',
        default=0,
        type=int,
        help='Number of gradient steps with only optimizing over rotations.'
    )

    parser.add_argument(
        '--TTT_only_motCorrupt_grad',
        default=False,
        action='store_true',
        help='Whether to only optimize over motion parameters in the motion corruption step.'
    )

    parser.add_argument(
        '--TTT_motState_batchSize_per_backprop',
        default=None,
        type=int,
        help='Batch size for each motion state during backpropagation.'
    )

    parser.add_argument(
        '--TTT_use_nufft_with_dcomp',
        default=False,
        action='store_true',
        help='Whether to use adjont NUFFT with density compensation at network input during TTT.'
    )

    parser.add_argument(
        '--TTT_nufft_max_coil_size',
        default=None,
        type=int,
        help='Maximum coil size for NUFFT during TTT. None means all coils are processed in one batch.'
    )

    parser.add_argument(
        '--args.TTT_use_clamp_schedule = False',
        default=False,
        action='store_true',
        help='Whether to use a schedule for clamping the motion parameters.'
    )

    parser.add_argument(
        '--eval_modeTTT_gtN_or_not',	
        default="",
        choices=("_gtN", ""),
        type=str,
        help='Whether to safe TTT figures with gtNorm scores or not and whether\
            to save best ssim recon over TTT epochs w.r.t. grNorm scores or not.'
    )

    parser.add_argument(
        '--list_of_samples_TTT',
        default=None,
        type=str,
        help='list of filenmes with slice numbers to use for TTT'
    )

    parser.add_argument(
        '--TTT_optimizer',
        default='Adam',
        choices=('SGD', 'Adam'),
        type=str,
        help='Optimizer for TTT.'
    )

    parser.add_argument(
        '--TTT_lr_max_decays',
        default=0,
        type=int,
        help='Number of learning rate decays for TTT.'
    )

    parser.add_argument(
        '--TTT_lr_decay_factor',
        default=0.1,
        type=float,
        help='Learning rate decay factor for TTT.'
    )

    parser.add_argument(
        '--TTT_lr_decay_after',
        default=20,
        type=int,
        help='Decay learning rate after this many epochs being smaller than the initial loss.'
    )

    parser.add_argument(
        '--TTT_lr_decay_at_the_latest',
        default=30,
        type=int,
        help='Decay learning rate at the latest after this many epochs.'
    )


    return parser

def get_optimizer_args(parser):

    parser.add_argument(
        '--optimizer',
        default="adam",
        choices=("adam"),
        type=str,
    )

    parser.add_argument(
        '--num_epochs',
        default=50,   
        type=int,
        help='Number of training epochs.'
    )

    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help='Learning rate.'
    )

    parser.add_argument(
        '--clip_grad_norm',
        default=False,
        action='store_true',
        help='Whether to clip gradient norm.'
    )

    parser.add_argument(
        '--scheduler',
        default='None',
        choices=('None', 'multistep', 'cosine'),
        type=str,
        help='Learning rate scheduler for training.'
    )

    parser.add_argument(
        '--multistep_milestones',
        default=[50, 60],
        type=int,
        nargs='+',
        help='Milestones for multistep scheduler.'
    )

    parser.add_argument(
        '--multistep_gamma',
        default=0.1,
        type=float,
        help='Gamma for multistep scheduler.'
    )

    parser.add_argument(
        '--cosine_min_lr',
        default=1e-6,
        type=float,
        help='Minimum learning rate for cosine scheduler.'
    )

    return parser

def get_model_args(parser):

    parser.add_argument(
        '--model',
        default="unet",
        choices=("unet"),
        type=str,
        help='The choice of model.'
    )

    parser.add_argument(
        '--load_model_from_checkpoint',
        default="None",
        type=str,
        help='Name of checkpoint to load model from.'
    )

    parser.add_argument(
        '--load_external_model_path',
        default="None",
        type=str,
        help='Path to external model to load.'
    )

    parser.add_argument(
        '--load_1ch_model',
        default=False,
        action='store_true',
        help='Whether to load a 1ch model and integrate its weights into a 2ch model.'
    )

    parser = get_unet_args(parser)

    return parser

def get_unet_args(parser):

    parser.add_argument(
        '--in_chans',
        default=2,
        type=int,
        help='Number of input channels for U-Net.'
    )

    parser.add_argument(
        '--out_chans',
        default=2,
        type=int,
        help='Number of output channels for U-Net.'
    )

    parser.add_argument(
        '--pools',
        default=4,
        type=int,
        help='Number of pooling layers for U-Net.'
    )

    parser.add_argument(
        '--chans',
        default=32,
        type=int,
        help='Number of top-level channels for U-Net.'
    )    

    return parser


def get_training_args(parser):

    parser.add_argument(
        '--train_loss',
        default="joint",
        choices=("sup_mag", "sup_ksp", "sup_compimg", "joint"),
        type=str,
        help='The choice of training loss.'
    )

    parser.add_argument(
        '--save_metrics',
        default=["SSIM", "PSNR"],
        nargs='+',
        type=str,
        help='Save model checkpoints at best metrics listed here.'
    )

    parser.add_argument(
        '--log_to_tb',
        default=True,
        action='store_false',
        help='Whether to log to tensorboard.'
    )

    parser.add_argument(
        '--val_every',
        default=1,
        type=int,
        help='Validate every this many epochs.'
    )

    parser.add_argument(
        '--log_imgs_to_tb_every',
        default=5,
        type=int,
        help='Log images to tensorboard every this many epochs.'
    )

    parser.add_argument(
        '--save_checkpoints_every',
        default=20,
        type=int,
        help='Save checkpoints every this many epochs.'
    )

    parser.add_argument(
        '--train_batch_size_per_axis',
        default=[10, 10, 10],
        type=int,
        nargs='+',
        help='Batch size per axis for training. [coronal, sagittal, axial]. Set [10, None, None] to only train one view.'
    )

    parser.add_argument(
        '--train_on_motion_corrected_inputs',
        default=False,
        action='store_true',
        help='Whether to train on motion corrected inputs.'
    )

    parser.add_argument(
        '--train_on_motion_corrupted_inputs',
        default=False,
        action='store_true',
        help='Whether to train on motion corrupted inputs.'
    )

    parser.add_argument(
        '--train_use_nufft_adjoint',
        default=False,
        action='store_true',
        help='Whether to use NUFFT adjoint to compute coarse recon at model input.'
    )

    parser.add_argument(
        '--train_use_nufft_with_dcomp',
        default=False,
        action='store_true',
        help='Whether to use adjont NUFFT with density compensation at network input during TTT.'
    )

    parser.add_argument(
        '--train_max_rots',
        default=[4,6],
        type=int,
        nargs='+',
        help='Maximum rotations during training are drawn randomly from this list.'
    )

    parser.add_argument(
        '--train_max_trans',
        default=[4,6],
        type=int,
        nargs='+',
        help='Maximum translations during training are drawn randomly from this list. Currently trans=rot.'
    )

    parser.add_argument(
        '--train_Ns',
        default=[10],
        type=int,
        nargs='+',
        help='Number of motion states during training are drawn randomly from this list.'
    )

    parser.add_argument(
        '--train_num_random_motion_seeds',
        default=1,
        type=int,
        help='Random seed for random motion during training are drawn randomly from np.arange(train_num_random_motion_seeds).'
    )

    parser.add_argument(
        '--train_one_grad_step_per_image_in_batch',
        default=False,
        action='store_true',
        help='Whether to perform one gradient step per image in batch.'
    )

    parser.add_argument(
        '--train_only_first_last_layers',
        default=False,
        action='store_true',
        help='Whether to train only first and last layers.'
    )

    parser.add_argument(
        '--train_load_optimizer_scheduler',
        default=False,
        action='store_true',
        help='Whether to load optimizer and scheduler from checkpoint.'
    )

    return parser

def get_data_args(parser):

    parser.add_argument(
        '--train_data_path',
        default="",
        type=str,
        help='Path to training data.'
    )
    parser.add_argument(
        '--train_set_path',
        default="",
        type=str,
        help='Path to pickel file with list of training data.'
    )
    parser.add_argument(
        '--train_mask_path',
        default="",
        type=str,
        help='Path to training mask.'
    )
    parser.add_argument(
        '--train_sensmaps_path',
        default="",
        type=str,
        help='Path to precomputed sensitivity maps.'
    )

    parser.add_argument(
        '--val_data_path',
        default="",
        type=str,
        help='Path to validation data.'
    )
    parser.add_argument(
        '--val_set_path',
        default="",
        type=str,
        help='Path to pickel file with list of validation data.'
    )
    parser.add_argument(
        '--val_mask_path',
        default="",
        type=str,
        help='Path to validation mask.'
    )
    parser.add_argument(
        '--val_sensmaps_path',
        default="",
        type=str,
        help='Path to precomputed sensitivity maps.'
    )

    parser.add_argument(
        '--TTT_example_path',
        default="",
        type=str,
        help='Path to the specific example we want to apply TTT to.'
    )
    parser.add_argument(
        '--TTT_mask_path',
        default="",
        type=str,
        help='Path to TTT mask.'
    )
    parser.add_argument(
        '--TTT_sensmaps_path',
        default="",
        type=str,
        help='Path to precomputed sensitivity maps.'
    )

    parser.add_argument(
        '--finalTestset_data_path',
        default="",
        type=str,
        help='Path to final test set data.'
    )

    parser.add_argument(
        '--load_data_to_gpu',
        default=False,
        action='store_true',
        help='Whether to load the data to GPU prior to training/testing.'
    )

    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help='Number of workers for dataloaders.'
    )

    parser.add_argument(
        '--pin_memory',
        default=False,
        action='store_true',
        help='Whether to pin memory for dataloaders.'
    )

    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size.'
    )

    parser.add_argument(
        '--data_drive',
        default="",
        type=str,
        help='Path to data drive.'
    )

    parser.add_argument(
        '--save_exp_results_path',
        default="",
        type=str,
        help='Path to save experiment results.'
    )

    return parser