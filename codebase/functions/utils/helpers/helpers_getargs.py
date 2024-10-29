from argparse import ArgumentParser

def get_args():

    parser = ArgumentParser()

    parser.add_argument(
        '--experiment_run_folder_name',
        default="",
        type=str,
        help='Name of folder to save experiment results. Must end with /'
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
    parser = get_L1min_args(parser)

    args = parser.parse_args(args=[])

    return args

def get_L1min_args(parser):

    parser.add_argument(
        '--L1min',
        default=False,
        action='store_true',
        help='Whether to apply L1-minimization based reconstruction.'
    )

    parser.add_argument(
        '--L1min_vivo',
        default=False,
        action='store_true',
        help='Whether to apply L1-minimization based reconstruction on in-vivo data.'
    )

    parser.add_argument(
        '--experiment_name_L1min',
        default="_",
        type=str,
        help='Name of L1min experiment.'
    )

    parser.add_argument(
        '--L1min_mode',
        default='gt_motion',
        choices=('gt_motion', 'pred_mot_motionTTT', 'pred_mot_altopt', 'noMoCo'),
        type=str,
        help='Which motion parameters to use for L1-minimization based reconstruction.'
    )

    parser.add_argument(
        '--L1min_on_TTT_load_from_phase',
        default=0,
        choices=(0,1,2),
        type=int,
        help='Which phase to load from TTT experiment.'
    )

    parser.add_argument(
        '--L1min_optimizer',
        default='Adam',
        choices=('SGD', 'Adam'),
        type=str,
        help='Optimizer for L1-minimization based reconstruction module.'
    )

    parser.add_argument(
        '--L1min_lr',
        default=1e-3,
        type=float,
        help='Learning rate for L1-minimization based reconstruction module.'
    )

    parser.add_argument(
        '--L1min_lambda',
        default=1e-3,
        type=float,
        help='Weight for L1 regularization during L1-minimization based reconstruction module.'
    )

    parser.add_argument(
        '--L1min_num_steps',
        default=50,
        type=int,
        help='Number of steps for L1-minimization based reconstruction module.'
    )

    parser.add_argument(
        '--L1min_nufft_max_coil_size',
        default=None,
        type=int,
        help='Maximum coil size for NUFFT during L1min. None means all coils are processed in one batch.'
    )

    parser.add_argument(
        '--L1min_DC_loss_thresholding',
        default=False,
        action='store_true',
        help='Whether to apply DC loss thresholding, i.e. ommit states with larger dc loss before L1min.'
    )

    parser.add_argument(
        '--L1min_DC_threshold',
        default=0.65,
        type=float,
        help='Threshold for dc loss.'
    )

    parser.add_argument(
        '--L1min_motion_alignment',
        default=False,
        action='store_true',
        help='Whether to align motion parameters before L1min.'
    )

    return parser

def get_alt_opt_args(parser):

    parser.add_argument(
        '--experiment_name_alt_opt',
        default="placeholder",
        type=str,
        help='Name of alt opt experiment. E.g., _lr001'
    )

    parser.add_argument(
        '--is_altopt_threshold',
        default=False,
        type=bool,
        help='Whether using the early stopping for the altopt baseline'
    )

    parser.add_argument(
        '--altopt_threshold',
        default=13,
        type=float,
        help='Threshold for the early stopping.(in log scale)'
    )

    parser.add_argument(
        '--alt_opt',
        default=False,
        action='store_true',
        help='Whether to apply alternating optimization.'
    )

    parser.add_argument(
        '--alt_opt_vivo',
        default=False,
        action='store_true',
        help='Whether to apply alternating optimization on the in-vivo test set.'
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
        '--altopt_recon_only_with_motionKnowledge_discretized',
        default=False,
        action='store_true',
        help='Whether to apply only reconstruction within alternating optimization with motion knowledge discretized.'
    )

    parser.add_argument(
        '--altopt_recon_only_with_motionKnowledge_remove_intraMotion',
        default=False,
        action='store_true',
        help='Whether to apply only reconstruction within alternating optimization with motion knowledge and remove intra-shot motion.'
    )

    parser.add_argument(
        '--alt_opt_on_TTTexp',
        default=False,
        action='store_true',
        help='Whether to load motion from TTT experiment and save results into TTT exp.'
    )

    parser.add_argument(
        '--alt_opt_on_TTT_load_from_phase',
        default=0,
        type=int,
        help='Which phase to load from TTT experiment.'
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
        default=0.65,
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
        default=50,
        type=int,
        help='Number of motion states (shots). First state always has no motion.'
    )

    parser.add_argument(
        '--num_shots',
        default=50,
        type=int,
        help='Number of shots.'
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

    parser.add_argument(
        '--center_in_first_state',
        default=False,
        action='store_true',
        help='Whether to center the object in the first state.'
    )

    parser.add_argument(
        '--fix_mot_maxksp_shot',
        default=False,
        action='store_true',
        help='Whether to fix (not learn) the motion for the max ksp shot and set to gt for sim motion.'
    )

    parser.add_argument(
        '--num_intraShot_events',
        default=0,
        type=int,
        help='Number of intra-shot motion events.'
    )

    parser.add_argument(
        '--sampTraj_simMot',
        default='interleaved_cartesian',
        type=str,
        choices=('random_cartesian', 'interleaved_cartesian', 'linear_cartesian', 'deterministic_cartesian'),
        help='Simulated sampling trajectory for motion simulation.'
    )

    parser.add_argument(
        '--random_sampTraj_seed',
        default=0,
        type=int,
        help='Random seed for random sampling trajectory.'
    )

    parser.add_argument(
        '--sampling_order_path',
        default=None,
        type=str,
        help='Path to sampling order for TTT.'
    )

    return parser

def get_TTT_args(parser):

    parser.add_argument(
        '--experiment_name_TTT',
        default="",
        type=str,
        help='Name of TTT experiment. E.g., _onlyTransl'
    )

    parser.add_argument(
        '--TTT',
        default=False,
        action='store_true',
        help='Whether to apply TTT to the model on the distribution-shift specific test set.'
    )

    parser.add_argument(
        '--TTT_vivo',
        default=False,
        action='store_true',
        help='Whether to apply TTT to the model on the in-vivo test set.'
    )

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
        '--TTT_motState_batchSize_per_backprop',
        default=None,
        type=int,
        help='Batch size for each motion state during backpropagation.'
    )

    parser.add_argument(
        '--TTT_use_nufft_with_dcomp',
        default=True,
        action='store_false',
        help='Whether to use adjont NUFFT with density compensation at network input during TTT.'
    )

    parser.add_argument(
        '--TTT_nufft_max_coil_size',
        default=None,
        type=int,
        help='Maximum coil size for NUFFT during TTT. None means all coils are processed in one batch.'
    )

    parser.add_argument(
        '--TTT_use_clamp_schedule',
        default=False,
        action='store_true',
        help='Whether to use a schedule for clamping the motion parameters.'
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
        '--TTT_lr_decay_at_the_latest',
        default=30,
        type=int,
        help='Decay learning rate at the latest after this many epochs.'
    )

    parser.add_argument(
        '--TTT_states_per_split',
        default=2,
        type=int,
        help='Split a state in this number of new states.'
    )

    parser.add_argument(
        '--TTT_list_of_split_steps',
        default=[],
        type=int,
        nargs='+',
        help='List of steps at which to split motion parameters.'
    )

    parser.add_argument(
        '--TTT_all_states_grad_after_split',
        default=False,
        action='store_true',
        help='Whether to optimize over all states after splitting.'
    )

    parser.add_argument(
        '--TTT_DCloss_th_split',
        default=0.575,
        type=float,
        help='Threshold for DC loss to split states after phase 0.'
    )

    parser.add_argument(
        '--TTT_lr_after_split',
        default=1.0,
        type=float,
        help='Learning rate after splitting.'
    )

    parser.add_argument(
        '--TTT_optimize_all_states_after',
        default=None,
        type=int,
        help='Optimize over all states after this many steps.'
    )

    parser.add_argument(
        '--TTT_optimize_all_states_after_lr',
        default=0.25,
        type=float,
        help='Learning rate for optimizing over all states during last steps (phase 3).'
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
        choices=("unet","stackedUnet"),
        type=str,
        help='The choice of model.'
    )

    parser.add_argument(
        '--load_model_path',
        default="None",
        type=str,
        help='Path to a local model to load.'
    )

    parser.add_argument(
        '--load_model_from_huggingface',
        default="None",
        type=str,
        help='Name of model to load from huggingface, e.g. mli-lab/Unet48-2D-CC359.'
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
        default=48,
        type=int,
        help='Number of top-level channels for U-Net.'
    )    

    return parser


def get_training_args(parser):

    parser.add_argument(
        '--experiment_name_train',
        default="E999_test",
        type=str,
        help='Name of experiment. Specifies the model checkpoint to load for TTT.'
    )

    parser.add_argument(
        '--test_run',
        default=False,
        action='store_true',
        help='Test run: 2 examples per dataset, 2 epochs.'
    )

    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='Whether to train the model.'
    )

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
        '--train_always_on_mild_motion',
        default=False,
        action='store_true',
        help='Whether to always train on mild motion corrupted inputs.'
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
        '--train_load_optimizer_scheduler',
        default=False,
        action='store_true',
        help='Whether to load optimizer and scheduler from checkpoint.'
    )

    parser.add_argument(
        '--eval_valset',
        default=False,
        action='store_true',
        help='Whether to perform a separate evaluation of an already trained model on the validation set.'
    )

    return parser

def get_data_args(parser):

    parser.add_argument(
        '--train_data_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to training data.'
    )
    parser.add_argument(
        '--train_set_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to pickel files with list of training data.'
    )
    parser.add_argument(
        '--train_mask_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to training mask.'
    )
    parser.add_argument(
        '--train_sensmaps_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to precomputed sensitivity maps.'
    )

    parser.add_argument(
        '--val_data_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to validation data.'
    )
    parser.add_argument(
        '--val_set_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to pickel files with list of validation data.'
    )
    parser.add_argument(
        '--val_mask_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to validation masks.'
    )
    parser.add_argument(
        '--val_sensmaps_paths',
        default=[""],
        type=str,
        nargs='+',
        help='List of paths to precomputed sensitivity maps.'
    )

    parser.add_argument(
        '--example_path',
        default="",
        type=str,
        help='Path to the specific example we want to apply TTT/altopt/L1min to.'
    )

    parser.add_argument(
        '--mask_path',
        default="",
        type=str,
        help='Path to undersampling mask.'
    )
    parser.add_argument(
        '--sensmaps_path',
        default="",
        type=str,
        help='Path to precomputed sensitivity maps.'
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

    return parser