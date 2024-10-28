from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
import torch


from functions.utils.helpers.helpers_math import complex_conj, complex_mul, complex_abs_sq
from functions.utils.helpers.helpers_math import ifft2c_ndim


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

class UnetDataTransform_Volume_fixMask:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        loaded_masks_dict: Dict,
    ):
        """
        Args:
            mask: A fixed mask array
        """
        self.loaded_masks_dict = loaded_masks_dict
    def __call__(
        self,
        kspace,
        sens_maps,
        datasource,
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

        mask = self.loaded_masks_dict[datasource]
        input_kspace = kspace * mask + 0.0

        target_img_3d = complex_mul(ifft2c_ndim(kspace, 3), sens_maps_conj).sum(dim=0, keepdim=False)

        return input_kspace, binary_background_mask, sens_maps_conj, target_img_3d, mask
    

