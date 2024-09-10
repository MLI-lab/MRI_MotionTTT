
import torch
#import pickle5 as pickle
import pickle
import logging
import os

from functions.data.transforms import UnetDataTransform_fixMask, UnetDataTransform_Volume_fixMask
from functions.data.mri_dataset import SliceDataset, VolumeDataset



def get_dataloader(args, mode):

    if mode == "TTT":
        data_sets = args.TTT_set_paths
        data_paths = args.TTT_data_paths
        mask_paths = args.TTT_mask_paths
        sensmaps_paths = args.TTT_sensmaps_paths
    elif mode == "train":
        data_sets = args.train_set_paths
        data_paths = args.train_data_paths
        mask_paths = args.train_mask_paths
        sensmaps_paths = args.train_sensmaps_paths
    elif mode == "val":
        data_sets = args.val_set_paths
        data_paths = args.val_data_paths
        mask_paths = args.val_mask_paths
        sensmaps_paths = args.val_sensmaps_paths


    # # Load all masks required for training or validation
    loaded_masks_dict = {}
    for key in mask_paths.keys():
        mask_path = mask_paths[key]

        with open(os.path.join(args.data_drive, mask_path),'rb') as fn:
            mask = pickle.load(fn)
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1) 
            logging.info(f"Using mask from {mask_path}")
        if args.load_data_to_gpu:
            mask = mask.cuda(args.gpu)
        loaded_masks_dict[key] = mask


    # # Define data transform and pass masks to it
    if args.model == "unet" and (args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint):
        data_transform = UnetDataTransform_Volume_fixMask(loaded_masks_dict, args)
    # Slice training currently not supported
    # else:
    #     data_transform = UnetDataTransform_fixMask(mask)


    num_samples = 2 if args.test_run else None
    if mode=="TTT" or mode=="val":
        if args.num_test_examples is not None:
            num_samples = args.num_test_examples

    if args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint:
        dataset = VolumeDataset(
            datasets = data_sets,
            data_paths = data_paths,
            path_to_sensmaps = sensmaps_paths,
            args=args,
            transform = data_transform,
            num_samples=num_samples,
        )
    # Slice training currently not supported
    # else:        
    #     dataset = SliceDataset(dataset = data_set,
    #                             data_path = data_path,
    #                             path_to_sensmaps = sensmaps_path, 
    #                             args=args,
    #                             transform = data_transform,
    #                             num_samples=num_samples,
    #                             )

    shuffle = True if mode == "train" else False
    dataloader = torch.utils.data.DataLoader(dataset = dataset, 
                                             batch_size = args.batch_size, 
                                             shuffle = shuffle, 
                                             pin_memory = args.pin_memory,
                                             num_workers=args.num_workers)
    
    return dataloader