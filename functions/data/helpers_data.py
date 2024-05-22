
import torch
#import pickle5 as pickle
import pickle
import logging
import os

from functions.data.transforms import UnetDataTransform_fixMask, UnetDataTransform_Volume_fixMask
from functions.data.mri_dataset import SliceDataset, VolumeDataset



def get_dataloader(args, mode):

    if mode == "TTT":
        data_set = args.TTT_set_path
        data_path = args.TTT_data_path
        mask_path = args.TTT_mask_path
        sensmaps_path = args.TTT_sensmaps_path
    elif mode == "train":
        data_set = args.train_set_path
        data_path = args.train_data_path
        mask_path = args.train_mask_path
        sensmaps_path = args.train_sensmaps_path
    elif mode == "val":
        data_set = args.val_set_path
        data_path = args.val_data_path
        mask_path = args.val_mask_path
        sensmaps_path = args.val_sensmaps_path


    with open(os.path.join(args.data_drive, mask_path),'rb') as fn:
        mask = pickle.load(fn)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1) 
        logging.info(f"Using mask from {mask_path}")
    if args.load_data_to_gpu:
        mask = mask.cuda(args.gpu)

    if args.model == "unet" and mode == "train" and (args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint):
        data_transform = UnetDataTransform_Volume_fixMask(mask, args)
        #logging.info("Using motion corrected inputs during training")
    elif mode == "val" and (args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint):
        data_transform = UnetDataTransform_Volume_fixMask(mask, args)
    else:
        data_transform = UnetDataTransform_fixMask(mask)


    num_samples = 2 if args.test_run else None
    if mode=="TTT" or mode=="val":
        if args.num_test_examples is not None:
            num_samples = args.num_test_examples

    if mode == "train" and (args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint):
        dataset = VolumeDataset(
            dataset = data_set,
            data_path = data_path,
            path_to_sensmaps = sensmaps_path,
            args=args,
            transform = data_transform,
            num_samples=num_samples,
        )
    elif mode == "val" and (args.train_on_motion_corrected_inputs or args.train_on_motion_corrupted_inputs or args.train_use_nufft_adjoint):
        dataset = VolumeDataset(
            dataset = data_set,
            data_path = data_path,
            path_to_sensmaps = sensmaps_path,
            args=args,
            transform = data_transform,
            num_samples=num_samples,
        )
    else:        
        dataset = SliceDataset(dataset = data_set,
                                data_path = data_path,
                                path_to_sensmaps = sensmaps_path, 
                                args=args,
                                transform = data_transform,
                                num_samples=num_samples,
                                )

    shuffle = True if mode == "train" else False
    dataloader = torch.utils.data.DataLoader(dataset = dataset, 
                                             batch_size = args.batch_size, 
                                             shuffle = shuffle, 
                                             pin_memory = args.pin_memory,
                                             num_workers=args.num_workers)
    
    return dataloader