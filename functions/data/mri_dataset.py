import os
#import pickle5 as pickle
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import torch
import numpy as np

from functions.helpers.helpers_log_save_image_utils import save_figure



class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    To access a single slice it loads the a triple of fully smapled kspace, sensmaps, and zerofilled image.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        path_to_sensmaps: str,
        args,
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset: Path to a file that contains a list of volumes/slices in the dataset.
            data_path: Path to a all the volumes/slices in the dataset.
            path_to_sensmaps: Path to a all the sensmaps. One sensmap for each slice
            args: The arguments of the experiment
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            num_samples: Optional; The number of samples to load from the dataset.
        """
        
        self.dataset = dataset
        self.path_to_sensmaps = path_to_sensmaps
        self.data_path = data_path
        self.args = args
        self.num_samples = num_samples
        self.transform = transform

        self.examples = []

        with open(os.path.join(args.data_drive, dataset),'rb') as fn:
            examples_list_of_dicts = pickle.load(fn)

        for example in examples_list_of_dicts:
            #filepath = os.path.join(args.data_drive,data_path,example["filename"])
            if example["axis"] == "Coronal":
                prefix = "cor"
            elif example["axis"] == "Sagittal":
                prefix = "sag"
            elif example["axis"] == "Axial":
                prefix = "ax"
            filepath = os.path.join(args.data_drive,data_path, f"{prefix}_s{example['position']}_" + example["filename"])
            axis = example["axis"]
            slice_num = example["position"]
            filename = example["filename"]
            self.examples.append((filepath, axis, slice_num, filename))

        # Take a subset of the datasets for fast trouble shooting
        if num_samples == None: 
            pass
        else: 
            self.examples = self.examples[0:num_samples]

        
        if self.args.load_data_to_gpu:
            # not implemented
            raise NotImplementedError
               

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        if self.args.load_data_to_gpu:
            raise NotImplementedError

        else:
            filepath, axis, slice_num, filename = self.examples[i]
        
            with h5py.File(filepath, "r") as hf:
                kspace = hf["kspace"][()]
                sens_maps = hf["smap"][()]
                inputs_img_full_2c = hf["zerofilled_img"][()]

            kspace = torch.from_numpy(kspace)
            sens_maps = torch.from_numpy(sens_maps)
            inputs_img_full_2c = torch.from_numpy(inputs_img_full_2c)
            sample = self.transform(kspace, sens_maps, inputs_img_full_2c, filename, axis, slice_num)

        return sample



class SliceDataset_loadVolume(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    To access a single slice it always loads the entire volume.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        path_to_sensmaps: str,
        args,
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset: Path to a file that contains a list of volumes/slices in the dataset.
            data_path: Path to a all the volumes/slices in the dataset.
            path_to_sensmaps: Path to a all the sensmaps. One sensmap for each slice
            args: The arguments of the experiment
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            num_samples: Optional; The number of samples to load from the dataset.
        """
        
        self.dataset = dataset
        self.path_to_sensmaps = path_to_sensmaps
        self.data_path = data_path
        self.args = args
        self.num_samples = num_samples
        self.transform = transform

        self.examples = []

        with open(os.path.join(args.data_drive, dataset),'rb') as fn:
            examples_list_of_dicts = pickle.load(fn)

        for example in examples_list_of_dicts:
            filepath = os.path.join(args.data_drive,data_path,example["filename"])
            axis = example["axis"]
            slice_num = example["position"]
            filename = example["filename"]
            self.examples.append((filepath, axis, slice_num, filename))

        # Take a subset of the datasets for fast trouble shooting
        if num_samples == None: 
            pass
        else: 
            self.examples = self.examples[0:num_samples]

        
        if self.args.load_data_to_gpu:
            self.examples_gpu=[]
            count=0
            for example in self.examples:
                print(count)
                count+=1
                filepath, axis, slice_num, filename = example

                smap_file = os.path.join(self.args.data_drive, self.path_to_sensmaps, "smaps_"+filename)
                with h5py.File(smap_file, 'r') as hf:
                    sens_maps = hf['smaps'][()]
                    sens_maps = torch.from_numpy(sens_maps).cuda(self.args.gpu)
           
                with h5py.File(filepath, "r") as hf:
                    kspace = hf["kspace"][()]
                    kspace = torch.from_numpy(kspace).cuda(self.args.gpu)

                self.examples_gpu.append((kspace, sens_maps, filename, axis, slice_num))
               

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        if self.args.load_data_to_gpu:
            kspace, sens_maps, filename, axis, slice_num = self.examples_gpu[i]

            sample = self.transform(kspace, sens_maps, filename, axis, slice_num)

        else:
            filepath, axis, slice_num, filename = self.examples[i]

            smap_file = os.path.join(self.args.data_drive, self.path_to_sensmaps, "smaps_"+filename)
            with h5py.File(smap_file, 'r') as hf:
                sens_maps = hf['smaps'][()]
        
            with h5py.File(filepath, "r") as hf:
                kspace = hf["kspace"][()]

            kspace = torch.from_numpy(kspace)
            sens_maps = torch.from_numpy(sens_maps)
            sample = self.transform(kspace, sens_maps, filename, axis, slice_num)

        return sample
    

class VolumeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image volumes.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        path_to_sensmaps: str,
        args,
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset: Path to a file that contains a list of volumes in the dataset.
            data_path: Path to a all the volumes in the dataset.
            path_to_sensmaps: Path to a all the sensmaps. One 3D sensmap for each volume
            args: The arguments of the experiment
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            num_samples: Optional; The number of samples to load from the dataset.
        """
        
        self.dataset = dataset
        self.path_to_sensmaps = path_to_sensmaps
        self.data_path = data_path
        self.args = args
        self.num_samples = num_samples
        self.transform = transform

        self.examples = []

        with open(os.path.join(args.data_drive, dataset),'rb') as fn:
            examples_list_of_dicts = pickle.load(fn)

        for example in examples_list_of_dicts:
            filepath = os.path.join(args.data_drive,data_path,example["filename"])
            filename = example["filename"]
            self.examples.append((filepath, filename))

        # Take a subset of the datasets for fast trouble shooting
        if num_samples == None: 
            pass
        else: 
            self.examples = self.examples[0:num_samples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 

        filepath, filename = self.examples[i]

        smap_file = os.path.join(self.args.data_drive, self.path_to_sensmaps, "smaps_"+filename)
        with h5py.File(smap_file, 'r') as hf:
            sens_maps_3D = hf['smaps'][()]
    
        with h5py.File(filepath, "r") as hf:
            kspace_3D = hf["kspace"][()]

        kspace_3D = torch.from_numpy(kspace_3D)
        sens_maps_3D = torch.from_numpy(sens_maps_3D)
        input_kspace, binary_background_mask_3D, sens_maps_conj_3D, target_img_3D, mask3D = self.transform(kspace_3D, sens_maps_3D)

        # for ax_ind, batch_size in enumerate(self.args.train_batch_size_per_axis):
        #     if batch_size:
        #         input_img_3D = input_img_3D.moveaxis(ax_ind, -2)
        #         rec_id = np.random.choice(range(input_img_3D.shape[2]),size=(batch_size), replace=False)

        return kspace_3D, input_kspace, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename
