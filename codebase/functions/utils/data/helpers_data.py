
import torch
import pickle
import logging

from functions.utils.data.transforms import UnetDataTransform_Volume_fixMask
from functions.utils.data.mri_dataset import VolumeDataset



def get_dataloader(args, mode):

    if mode == "train":
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

        with open(mask_path,'rb') as fn:
            mask = pickle.load(fn)
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1) 
            logging.info(f"Using mask from {mask_path}")
        if args.load_data_to_gpu:
            mask = mask.cuda(args.gpu)
        loaded_masks_dict[key] = mask

    # # Define data transform and pass masks to it
    data_transform = UnetDataTransform_Volume_fixMask(loaded_masks_dict)

    num_samples = 2 if args.test_run else None

    dataset = VolumeDataset(
        datasets = data_sets,
        data_paths = data_paths,
        path_to_sensmaps = sensmaps_paths,
        args=args,
        transform = data_transform,
        num_samples=num_samples,
    )

    shuffle = True if mode == "train" else False
    dataloader = torch.utils.data.DataLoader(dataset = dataset, 
                                             batch_size = 1, 
                                             shuffle = shuffle, 
                                             pin_memory = args.pin_memory,
                                             num_workers=args.num_workers)
    
    return dataloader