import torch
import sys
import functions.motion_simulation.interp as interp
from functions.helpers.helpers_math import *
# from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj


from math import ceil
import numpy as np
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex

def nufft_adjoint(input, coord, out_shape, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    out_shape = list(out_shape)

    os_shape = _get_oversamp_shape(out_shape, ndim, oversamp)

    # Gridding
    out_shape2 = out_shape.copy()
    os_shape2 = os_shape.copy()
    coord = _scale_coord(coord, out_shape2, oversamp, device)

    kernel = _get_kaiser_bessel_kernel(128, width, beta, coord.dtype, device)
    output = interp.gridding(input, os_shape2, width, kernel, coord, ndim, device)
#     print(output.shape)
    output = output/(width**ndim)
    
    # IFFT
    out_complex_form = convert_to_tensor_complex(output)
    out_complex_form   = ifft3(out_complex_form)
    output = torch.complex(out_complex_form[...,0],out_complex_form[...,1])

    # Crop
    output = resize(output, out_shape2, device=device)
    a = prod(os_shape2[-ndim:]) / prod(out_shape2[-ndim:]) ** 0.5
    output = output * a
    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)
    
    return output

def _apodize(input, ndim, oversamp, width, beta, device):
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = _get_ugly_number(oversamp * i)
        idx = torch.arange(i, device=device)

        # Calculate apodization
        apod = (beta ** 2 - (np.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1))

    return output

def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i) for i in shape[-ndim:]]

def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord

def _get_ugly_number(n):
    if n <= 1:
        return n

    ugly_nums = [1]
    i2, i3, i5 = 0, 0, 0
    while (True):

        ugly_num = min(ugly_nums[i2] * 2,
                       ugly_nums[i3] * 3,
                       ugly_nums[i5] * 5)

        if ugly_num >= n:
            return ugly_num

        ugly_nums.append(ugly_num)
        if ugly_num == ugly_nums[i2] * 2:
            i2 += 1
        elif ugly_num == ugly_nums[i3] * 3:
            i3 += 1
        elif ugly_num == ugly_nums[i5] * 5:
            i5 += 1

def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    x = torch.arange(n, dtype=dtype) / n
    kernel = 1 / width * torch.tensor(np.i0(beta * (1 - x ** 2) ** 0.5), dtype=dtype)
    return kernel.to(device)

def prod(shape):
    """Computes product of shape.
    Args:
        shape (tuple or list): shape.
    Returns:
        Product.
    """
    return np.prod(shape)

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None,device='cuda'):
    ishape_exp, oshape_exp = _expand_shapes(input.shape, oshape)

    if ishape_exp == oshape_exp:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    copy_shape = [min(i - si, o - so) for i, si, o,
                  so in zip(ishape_exp, ishift, oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape_exp, dtype=input.dtype, device=device)
    input = input.reshape(ishape_exp)
    output[oslice] = input[islice]

    return output.reshape(oshape)

def ifft3(data):
    return ifft2c_ndim(data, signal_ndim=3)