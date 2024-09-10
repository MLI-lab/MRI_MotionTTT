
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import scipy.stats as ss

from functions.helpers.helpers_math import complex_abs
#from functions.helpers.helpers_log_save_image_utils import save_figure

@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)

def random_mask_partioning_3D(mask3D,gpu):

    assert len(mask3D.shape) == 5, "we expect 5 dimensional mask with (1,x,y,z,1)"

    ### mask partitioning for self validation: we define a left-out set for early-stopping
    gen = torch.Generator(device=torch.device('cpu'))
    gen.manual_seed(1)
    where_ones = torch.where(mask3D==1) # gives tuple of tensors with number of entries equal to the number of dimensions of mask3D
    m = len(where_ones[0]) # number of elements that are 1
    random_indices = torch.randint(0, m, (m//20,), generator=gen, device=torch.device('cpu')).cuda(gpu)

    mask_TTT_fit_loss = mask3D.clone()
    mask_TTT_fit_loss[:,where_ones[1][random_indices],where_ones[2][random_indices],where_ones[3][random_indices],:] = 0.0

    mask_TTT_selfval_loss = mask3D.clone() * 0.0
    mask_TTT_selfval_loss[:,where_ones[1][random_indices],where_ones[2][random_indices],where_ones[3][random_indices],:] = 1.0

    assert torch.sum(mask_TTT_fit_loss + mask_TTT_selfval_loss) == torch.sum(mask3D), "something went wrong with the partitioning"
    assert torch.sum(mask_TTT_fit_loss * mask_TTT_selfval_loss) == 0, "something went wrong with the partitioning"
    return mask_TTT_fit_loss, mask_TTT_selfval_loss

class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This creates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    It creates one mask for the input and one mask for the target.
    """

    def __init__(self, center_fraction: float, acceleration: float):
        """
        Args:
            self_sup: If False the target mask is all ones. If True the target mask is also undersampled
            center_fractions: Fraction of low-frequency columns to be retained both in input and target.
            accelerations: Amount of under-sampling for the input
            acceleration_total: Required if self_sup=True. Determines how much measurements are available for the split into input and target masks
        """
        self.center_fraction = center_fraction #cent
        self.acceleration = acceleration #p

        self.rng = np.random.RandomState()  # pylint: disable=no-member


    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError
    

class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            
            center_fraction = self.center_fraction
            acceleration = self.acceleration
            num_cols = shape[-2]
            num_rows = shape[-3]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32) # initialize with zeros
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True # set center fractions to 1

            # due to dense center fraction we have to increase the acceleration s.t. in total only the correct amount of freqs are sampled
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols)

            # This random offset might be quite important because otherwise images of the
            # same dimesion would always mask exactly the same freqs
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols

            input_mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            # repeat mask according to num rows to get a 2d mask
            input_mask = input_mask.repeat(1,num_rows,1,1)



        return input_mask

def create_mask_for_mask_type(
    mask_type_str: str,
    #self_sup: bool,
    center_fraction: float,
    acceleration: float,
    #acceleration_total: Optional[float],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    #if mask_type_str == "random":
    #    return RandomMaskFunc(self_sup, center_fraction, acceleration, acceleration_total)
    #    #raise Exception(f"{mask_type_str} masks not implemented in this framework")
    if mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fraction, acceleration)
        #raise Exception(f"{mask_type_str} masks not implemented in this framework")
    #elif mask_type_str == "n2n":
    #    return n2nMaskFunc(self_sup, center_fraction, acceleration, acceleration_total)
    else:
        raise Exception(f"{mask_type_str} not supported")
    


def apply_mask(
    shape: np.ndarray,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        fix_selfsup_inputtarget_split: Only important for self-sup training. 
            If it is False the input/target split is random. Always True for validation and testing. 
            Determined by hp_exp['use_mask_seed_for_training'] during self-sup training.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """

    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros


    return mask


