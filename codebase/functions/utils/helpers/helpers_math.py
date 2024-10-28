import torch
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
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)

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

