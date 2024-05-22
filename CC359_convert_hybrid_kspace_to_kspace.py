
# %%
import torch
import h5py
import pathlib
import os

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

# %%

# %%
# Convert hybrid kspace into full-Fourier-domain kspace and save

input_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train/"

mri_files = list(pathlib.Path(input_dir).glob('*.h5'))
print(len(mri_files))
output_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/"

for mri_f in mri_files:

    with h5py.File(mri_f, 'r') as hf:
        kspace_hf = hf['kspace'][()]

    kspace_torch = torch.from_numpy(kspace_hf)
    kspace_torch_cpx = torch.stack([kspace_torch[...,::2], kspace_torch[...,1::2]], dim=-1).moveaxis(3,1)
    kspace_torch_cpx[:,:,:,144:,:] = 0.0
    kspace_torch_cpx = kspace_torch_cpx.moveaxis(0,-2)

    kspace_torch_cpx_3D = fft2c_ndim(kspace_torch_cpx, 1)
    img_torch_cpx_3D = ifft2c_ndim(kspace_torch_cpx_3D, 3)
    img_torch_cpx_3D_shifted = torch.fft.ifftshift(img_torch_cpx_3D, dim=(-4,-3))

    kspace_torch_cpx_3D_recon = fft2c_ndim(img_torch_cpx_3D_shifted, 3)
    kspace_torch_cpx_3D_recon[:,:,144:,:,:] = 0.0
    

    save_file = os.path.join(output_dir, str(mri_f.name))
    print(save_file)
    data = h5py.File(save_file, 'w')
    data.create_dataset('kspace', data=kspace_torch_cpx_3D_recon)
    data.close()


input_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val/"

mri_files = list(pathlib.Path(input_dir).glob('*.h5'))
print(len(mri_files))
output_dir = "/media/ssd3/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Val_converted/"

for mri_f in mri_files:

    with h5py.File(mri_f, 'r') as hf:
        kspace_hf = hf['kspace'][()]

    kspace_torch = torch.from_numpy(kspace_hf)
    kspace_torch_cpx = torch.stack([kspace_torch[...,::2], kspace_torch[...,1::2]], dim=-1).moveaxis(3,1)
    kspace_torch_cpx[:,:,:,144:,:] = 0.0
    kspace_torch_cpx = kspace_torch_cpx.moveaxis(0,-2)

    kspace_torch_cpx_3D = fft2c_ndim(kspace_torch_cpx, 1)
    img_torch_cpx_3D = ifft2c_ndim(kspace_torch_cpx_3D, 3)
    img_torch_cpx_3D_shifted = torch.fft.ifftshift(img_torch_cpx_3D, dim=(-4,-3))

    kspace_torch_cpx_3D_recon = fft2c_ndim(img_torch_cpx_3D_shifted, 3)
    kspace_torch_cpx_3D_recon[:,:,144:,:,:] = 0.0
    

    save_file = os.path.join(output_dir, str(mri_f.name))
    print(save_file)
    data = h5py.File(save_file, 'w')
    data.create_dataset('kspace', data=kspace_torch_cpx_3D_recon)
    data.close()

# %%

