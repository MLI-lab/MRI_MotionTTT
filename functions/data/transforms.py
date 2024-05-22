from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os

from functions.helpers.helpers_math import complex_abs, complex_conj, complex_mul, complex_abs_sq
from functions.helpers.helpers_math import fft2c_ndim, ifft2c_ndim

from functions.motion_simulation.motion_functions import motion_correction_NUFFT, generate_random_motion_params
from functions.motion_simulation.motion_functions import generate_interleaved_cartesian_trajectory, motion_corruption_NUFFT


#from functions.helpers.helpers_log_save_image_utils import save_figure



def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    # record time of following operation
    
    if np.iscomplexobj(data):
        #start_time = time.time()
        data = np.stack((data.real, data.imag), axis=-1)
        #print("Separate real from imag: %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    data = torch.from_numpy(data)
    #print("Convert to tensor: %s seconds ---" % (time.time() - start_time))
    return data





# test-time traing scale normalization

def rss_torch(im, coil_dim=0):
    '''
    Apply the root sum of squares algorithm to coil images
    im should be im=complex_abs(im_complex), if not this operation will keep the
    complex dims.
    '''
    return torch.sqrt(torch.sum(torch.abs(im) ** 2, coil_dim, keepdim=False))

def rss_complex(im: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        im: The input tensor. Equivalent to rss_troch if im=im_complex
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(im).sum(dim))



class UnetDataTransform_fixMask:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        mask: torch.Tensor,

    ):
        """
        Args:
            mask: A fixed mask array
        """
        self.mask = mask


    def __call__(
        self,
        target_kspace,
        sens_maps,
        inputs_img_full_2c,
        filename: str,
        axis: str,
        slice_num: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        """
        Args:
            target_kspace: Input k-space of shape (num_coils, x, y, 2)
            sens_maps: Sensitivity maps of shape shape (num_coils, x, y, 2)
            inputs_img_full_2c: Zero-filled input image of shape (x, y, 2)
            filename: File name for logging
            axis: Axis of the slice (Axial, Coronal, Sagittal)
            slice_num: slice number along axis

        Returns:
            tuple containing:
                inputs_img_full_2c: Zero-filled input image
                input_kspace: Undersampled input kspace
                target_image_full_2c: Complex ground truth image computed with sensitivity maps
                kspace: Ground truth kspace
                sens_maps: sensitivity maps to compute expand operations
                sens_maps_conj: Complex conjugate of sensitivity maps to compute reduce operations
                mask: 1d input mask
                mask2d: 2d input mask
                binary_background_mask: mask to apply before final score computation
                fname: File name for logging
                slice_num: Serial number of the slice for logging
                crop_size: Size for cropping images before final score computation and visualization

        """
        sens_maps_conj = complex_conj(sens_maps)

        binary_background_mask = torch.round(torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,0:1])

        # check if background of sensitivity maps is always excactly 0
        if torch.sum(torch.abs(torch.abs(binary_background_mask.unsqueeze(0)-1)*sens_maps)).item() != 0.0:
            raise ValueError("The background of the these sensitivity maps is not exactly 0.0!")

        set = torch.unique(binary_background_mask)
        if binary_background_mask.max() != 1.0 or binary_background_mask.min() != 0.0 or set.shape[0] != 2:
            if binary_background_mask.min() != 1.0:
                print(binary_background_mask.max(),binary_background_mask.min())
                print(filename, slice_num)
                for i in range(set.shape[0]):
                    print(set[i].item())
                raise ValueError("Warning: The real part of the sensitivity maps times their complex conjugate is not a binary mask!")


        target_image_full_2c = complex_mul(ifft2c_ndim(target_kspace, 2), sens_maps_conj).sum(dim=0, keepdim=False)


        return inputs_img_full_2c, target_image_full_2c, target_kspace, sens_maps, sens_maps_conj, binary_background_mask, filename, slice_num, axis


class UnetDataTransform_Volume_fixMask:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        args: Dict = None,
    ):
        """
        Args:
            mask: A fixed mask array
        """
        self.mask = mask
        self.args = args


    def __call__(
        self,
        kspace,
        sens_maps,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, phase encoding, slice encoding, frequency encoding, 2)
            sens_maps: Sensitivity maps of shape shape (num_coils, phase encoding, slice encoding, frequency encoding, 2)
        Returns:
            tuple containing:
                input_img_3d: Zero-filled input volume
                

        """
        sens_maps_conj = complex_conj(sens_maps)

        binary_background_mask = torch.round(torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,:,0:1])

        # check if background of sensitivity maps is always excactly 0
        if torch.sum(torch.abs(torch.abs(binary_background_mask.unsqueeze(0)-1)*sens_maps)).item() != 0.0:
            raise ValueError("The background of the these sensitivity maps is not exactly 0.0!")

        set = torch.unique(binary_background_mask)
        if binary_background_mask.max() != 1.0 or binary_background_mask.min() != 0.0 or set.shape[0] != 2:
            if binary_background_mask.min() != 1.0:
                print(binary_background_mask.max(),binary_background_mask.min())
                #print(filename, slice_num)
                for i in range(set.shape[0]):
                    print(set[i].item())
                raise ValueError("Warning: The real part of the sensitivity maps times their complex conjugate is not a binary mask!")

        input_kspace = kspace * self.mask + 0.0

        target_img_3d = complex_mul(ifft2c_ndim(kspace, 3), sens_maps_conj).sum(dim=0, keepdim=False)

        return input_kspace, binary_background_mask, sens_maps_conj, target_img_3d, self.mask
    



class UnetDataTransform_fromVolume_fixMask:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        mask: torch.Tensor,

    ):
        """
        Args:
            mask: A fixed mask array
        """
        self.mask = mask


    def __call__(
        self,
        kspace,
        sens_maps,
        filename: str,
        axis: str,
        slice_num: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, phase encoding, slice encoding, frequency encoding, 2)
            sens_maps: Sensitivity maps of shape shape (num_coils, phase encoding, slice encoding, frequency encoding, 2)
            filename: File name for logging
            axis: Axis of the slice (Axial, Coronal, Sagittal)
            slice_num: slice number along axis

        Returns:
            tuple containing:
                inputs_img_full_2c: Zero-filled input image
                input_kspace: Undersampled input kspace
                target_image_full_2c: Complex ground truth image computed with sensitivity maps
                kspace: Ground truth kspace
                sens_maps: sensitivity maps to compute expand operations
                sens_maps_conj: Complex conjugate of sensitivity maps to compute reduce operations
                mask: 1d input mask
                mask2d: 2d input mask
                binary_background_mask: mask to apply before final score computation
                fname: File name for logging
                slice_num: Serial number of the slice for logging
                crop_size: Size for cropping images before final score computation and visualization

        """
        sens_maps_conj = complex_conj(sens_maps)

        binary_background_mask = torch.round(torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,:,0:1])

        # check if background of sensitivity maps is always excactly 0
        if torch.sum(torch.abs(torch.abs(binary_background_mask.unsqueeze(0)-1)*sens_maps)).item() != 0.0:
            raise ValueError("The background of the these sensitivity maps is not exactly 0.0!")

        set = torch.unique(binary_background_mask)
        if binary_background_mask.max() != 1.0 or binary_background_mask.min() != 0.0 or set.shape[0] != 2:
            if binary_background_mask.min() != 1.0:
                print(binary_background_mask.max(),binary_background_mask.min())
                print(filename, slice_num)
                for i in range(set.shape[0]):
                    print(set[i].item())
                raise ValueError("Warning: The real part of the sensitivity maps times their complex conjugate is not a binary mask!")

        input_kspace = kspace * self.mask + 0.0
        

        input_img_volume = complex_mul(ifft2c_ndim(input_kspace, 3), sens_maps_conj).sum(dim=0, keepdim=False)

        if axis == "Axial":
            sens_maps_2d = sens_maps[:,:,:,slice_num,:]
            sens_maps_conj_2d = sens_maps_conj[:,:,:,slice_num,:]
            inputs_img_full_2c = input_img_volume[:,:,slice_num,:]
            kspace_hybrid = ifft2c_ndim(kspace, 1)
            target_kspace = kspace_hybrid[:,:,:,slice_num,:]
            target_image_full_2c = complex_mul(ifft2c_ndim(target_kspace, 2), sens_maps_conj_2d).sum(dim=0, keepdim=False)
            binary_background_mask = binary_background_mask[:,:,slice_num,:]
        elif axis == "Coronal":
            sens_maps_2d = sens_maps[:,slice_num,:,:,:]
            sens_maps_conj_2d = sens_maps_conj[:,slice_num,:,:,:]
            inputs_img_full_2c = input_img_volume[slice_num,:,:,:]
            kspace_hybrid = ifft2c_ndim(kspace.moveaxis(1,-2), 1)
            target_kspace = kspace_hybrid[:,:,:,slice_num,:]
            target_image_full_2c = complex_mul(ifft2c_ndim(target_kspace, 2), sens_maps_conj_2d).sum(dim=0, keepdim=False)
            binary_background_mask = binary_background_mask[slice_num,:,:,:]
        elif axis == "Sagittal":
            sens_maps_2d = sens_maps[:,:,slice_num,:,:]
            sens_maps_conj_2d = sens_maps_conj[:,:,slice_num,:,:]
            inputs_img_full_2c = input_img_volume[:,slice_num,:,:]
            kspace_hybrid = ifft2c_ndim(kspace.moveaxis(2,-2), 1)
            target_kspace = kspace_hybrid[:,:,:,slice_num,:]
            target_image_full_2c = complex_mul(ifft2c_ndim(target_kspace, 2), sens_maps_conj_2d).sum(dim=0, keepdim=False)
            binary_background_mask = binary_background_mask[:,slice_num,:,:]



        return inputs_img_full_2c, target_image_full_2c, target_kspace, sens_maps_2d, sens_maps_conj_2d, binary_background_mask, filename, slice_num, axis
    
