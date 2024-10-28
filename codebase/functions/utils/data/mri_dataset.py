import os
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import torch


class VolumeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image volumes.
    """

    def __init__(
        self,
        datasets: list,
        data_paths: str,
        path_to_sensmaps: str,
        args,
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            datasets: List of paths to a files that define the datasets.
            data_path: Dictionary containing one key for each dataset in datasets
                and as value the corresponding path the data in that dataset
            path_to_sensmaps: Dictionary containing one key for each dataset in datasets
                and as value the corresponding path the pre-computed 3D sensitivity maps
            args: The arguments of the experiment
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form.
            num_samples: Optional; The number of samples to load from the dataset.
        """
        
        self.datasets = datasets
        self.path_to_sensmaps = path_to_sensmaps
        self.data_paths = data_paths
        self.args = args
        self.num_samples = num_samples
        self.transform = transform

        self.examples = []

        for dataset in datasets:
            
            with open(dataset,'rb') as fn:
                examples_list_of_dicts = pickle.load(fn)

            for example in examples_list_of_dicts:
                datasource = example["datasouce"]
                datatype = example["datatype"]
                filename = example["filename"]
                if "motion_seeds" in example:
                    random_motion_seeds = example["motion_seeds"]
                else:
                    random_motion_seeds = 0
                data_path = data_paths[datasource]
                
                filepath = os.path.join(data_path,filename)
                
                self.examples.append((filepath, filename, datasource, datatype, random_motion_seeds))

        # Take a subset of the datasets for fast trouble shooting
        if num_samples == None: 
            pass
        else: 
            self.examples = self.examples[0:num_samples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 

        filepath, filename, datasource, datatype, random_motion_seeds = self.examples[i]

        path_to_sensmaps = self.path_to_sensmaps[datasource]

        smap_file = os.path.join(path_to_sensmaps, "smaps_"+filename)
        with h5py.File(smap_file, 'r') as hf:
            sens_maps_3D = hf['smaps'][()]
    
        with h5py.File(filepath, "r") as hf:
            kspace_3D = hf["kspace"][()]

        kspace_3D = torch.from_numpy(kspace_3D)
        sens_maps_3D = torch.from_numpy(sens_maps_3D)
        input_kspace, binary_background_mask_3D, sens_maps_conj_3D, target_img_3D, mask3D = self.transform(kspace_3D, sens_maps_3D, datasource)

        return kspace_3D, input_kspace, binary_background_mask_3D, sens_maps_3D, sens_maps_conj_3D, target_img_3D, mask3D, filename, random_motion_seeds
