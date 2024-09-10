"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch

import numpy as np
from typing import Tuple, Union

def chunks(l, n):
    """Yield n number of sequential chunks from l.
    E.g. list(chunks([0,1,2,3,4,5,6],3)) -> [[0,1,2],[3,4],[5,6]]
    
    """
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

def norm_to_gt(x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Normalize x to have the same mean and standard deviation as gt.

    Args:
        x: A PyTorch tensor.
        gt: A PyTorch tensor.

    Returns:
        A PyTorch tensor.
    """
    assert x.shape == gt.shape
    x = x - x.mean()
    x = x / x.std()
    x = x * gt.std()
    x = x + gt.mean()

    return x

def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]



########################
# Fourier transform


def fft2c_ndim(data, signal_ndim):
    """
    Apply centered 1/2/3-dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 2 dimensions:
            dimension -1 has size 2 containing real and imaginary part 
            dimensions -2 and potentially -3 and -4 are spatial dimensions 
            All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    if signal_ndim == 1:
        dims = (-1)
    elif signal_ndim == 2:
        dims = (-2, -1)
    elif signal_ndim == 3:
        dims = (-3, -2, -1)

    data_cpx = torch.view_as_complex(data)
    data_cpx = torch.fft.ifftshift(data_cpx, dim=dims)
    data_cpx = torch.fft.fftn(data_cpx, dim=dims, norm="ortho")
    data_cpx = torch.fft.fftshift(data_cpx, dim=dims)


    return torch.view_as_real(data_cpx)

def ifft2c_ndim(data, signal_ndim):
    """
    Apply centered 1/2/3-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 2 dimensions:
            dimension -1 has size 2 containing real and imaginary part 
            dimensions -2 and potentially -3 and -4 are spatial dimensions 
            All other dimensions are assumed to be batch dimensions.

    Returns:
        The IFFT of the data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    if signal_ndim == 1:
        dims = (-1)
    elif signal_ndim == 2:
        dims = (-2, -1)
    elif signal_ndim == 3:
        dims = (-3, -2, -1)

    data_cpx = torch.view_as_complex(data)
    data_cpx = torch.fft.ifftshift(data_cpx, dim=dims)
    data_cpx = torch.fft.ifftn(data_cpx, dim=dims, norm="ortho")
    data_cpx = torch.fft.fftshift(data_cpx, dim=dims)


    return torch.view_as_real(data_cpx)



########################
# Normalizatons

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_to_given_mean_std(
    im1: torch.Tensor,
    im2: torch.Tensor
    ) -> torch.Tensor:
    """
    This function computes the mean and std of im1 and normalizes im2 to have this mean and std.
    """
    im2 = (im2-im2.mean()) / im2.std()
    im2 *= im1.std()
    im2 += im1.mean()
    return im1,im2


def normalize_separate_over_ch(
    x: torch.Tensor,
    mean: Union[float, torch.Tensor] = None,
    std: Union[float, torch.Tensor] = None,
    eps: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    If mean and stddev is given x is normalized to have this mean and std.
    If not given x is normalized to have 0 mean and std 1.
    x is supposed to have shape b,c,h,w and normalization is only over h,w
    Hence mean and std have shape b,c,1,1
    """
    if x.shape[-1]==2:
        raise ValueError("Group normalize does not expect complex dim at last position.")
    if len(x.shape) != 4:
        raise ValueError("Gourp normalize expects four dimensions in the input tensor: (batch, ch, x, y)")

    # group norm
    if mean == None and std == None:
        mean = x.mean(dim=[2,3],keepdim=True)
        std = x.std(dim=[2,3],keepdim=True)

    return (x - mean) / (std + eps), mean, std

def normalize_separate_over_ch_3D(
    x: torch.Tensor,
    mean: Union[float, torch.Tensor] = None,
    std: Union[float, torch.Tensor] = None,
    eps: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    If mean and stddev is given x is normalized to have this mean and std.
    If not given x is normalized to have 0 mean and std 1.
    x is supposed to have shape b,c,h,w,d and normalization is only over h,w,d
    Hence mean and std have shape b,c,1,1,1
    """
    if x.shape[-1]==2:
        raise ValueError("Group normalize does not expect complex dim at last position.")
    if len(x.shape) != 5:
        raise ValueError("Gourp normalize expects four dimensions in the input tensor: (batch, ch, x, y, z)")

    # group norm
    if mean == None and std == None:
        mean = x.mean(dim=[2,3,4],keepdim=True)
        std = x.std(dim=[2,3,4],keepdim=True)

    return (x - mean) / (std + eps), mean, std


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


########################
# Cropping

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div(data.shape[-2] - shape[0], 2, rounding_mode='trunc') # (data.shape[-2] - shape[0]) // 2
    h_from = torch.div(data.shape[-1] - shape[1], 2, rounding_mode='trunc') # (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y