
# %%
import numpy as np
import torch
import h5py
import pathlib
import pickle
import os
import sys

### after install bart 0.7.00 from https://mrirecon.github.io/bart/, import it as follows
sys.path.insert(0,'/bart-0.7.00/python/')
os.environ['TOOLBOX_PATH'] = "/bart-0.7.00/"
import bart

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
    
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    data = torch.from_numpy(data)
    return data

# %%
# %%

load_path = "mask_3D_size_218_170_256_R_4_poisson_disc.pickle"
with open(load_path, 'rb') as handle:
    mask = pickle.load(handle)


input_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/"

mri_files = list(pathlib.Path(input_dir).glob('*.h5'))
print(len(mri_files))
output_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_s_maps_3D/"

for mri_f in mri_files:

    save_file = os.path.join(output_dir, "smaps_"+str(mri_f.name))
    if os.path.exists(save_file):
        continue

    with h5py.File(mri_f, 'r') as hf:
        kspace_hf = hf['kspace'][()]

    if kspace_hf.shape[2] != 170:
        continue
    kspace_hf = torch.from_numpy(kspace_hf)
    masked_k_space = kspace_hf * torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)

    masked_k_space_1c = masked_k_space[...,0] + 1j*masked_k_space[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', masked_k_space_1c.moveaxis(0,-1).numpy())

    sens_maps_torch = to_tensor(sens_maps).moveaxis(-2,0)

    
    print(save_file)
    data = h5py.File(save_file, 'w')
    data.create_dataset('smaps', data=sens_maps_torch)
    data.close()

input_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/"

mri_files = list(pathlib.Path(input_dir).glob('*.h5'))
print(len(mri_files))
output_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_s_maps_3D/"

for mri_f in mri_files:

    save_file = os.path.join(output_dir, "smaps_"+str(mri_f.name))
    if os.path.exists(save_file):
        continue

    with h5py.File(mri_f, 'r') as hf:
        kspace_hf = hf['kspace'][()]

    if kspace_hf.shape[2] != 170:
        continue

    kspace_hf = torch.from_numpy(kspace_hf)
    masked_k_space = kspace_hf * torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)

    masked_k_space_1c = masked_k_space[...,0] + 1j*masked_k_space[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', masked_k_space_1c.moveaxis(0,-1).numpy())

    sens_maps_torch = to_tensor(sens_maps).moveaxis(-2,0)

    print(save_file)
    data = h5py.File(save_file, 'w')
    data.create_dataset('smaps', data=sens_maps_torch)
    data.close()

# %%
