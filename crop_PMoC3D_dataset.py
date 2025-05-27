import os
import h5py
import torch
from codebase.functions.utils.helpers.helpers_math import complex_abs,ifft2c_ndim, fft2c_ndim,complex_conj,complex_mul
from tqdm import tqdm

def crop_kspace(kspace):
    data_cpx = torch.from_numpy(kspace).flip(2).moveaxis(-2, 1).flip(2)
    crop_freq_enc_to_size = 256
    # Crop the data on the frequency encoding direction:
    data_cpx = torch.view_as_complex(data_cpx)
    Nx = data_cpx.shape[1]
    data_cpx = torch.fft.ifftshift(data_cpx, dim=(1))
    data_cpx = torch.fft.ifftn(data_cpx, dim=(1), norm="ortho")
    data_cpx = torch.fft.fftshift(data_cpx, dim=(1))
    # Crop the data
    data_crop = data_cpx[:,(Nx-crop_freq_enc_to_size)//2:(Nx+crop_freq_enc_to_size)//2,...]
    # 1D back transform:
    data_crop = torch.fft.fftshift(data_crop, dim=(1))
    data_crop = torch.fft.fftn(data_crop, dim=(1), norm="ortho")
    data_crop = torch.fft.ifftshift(data_crop, dim=(1))
    kspace_crop = torch.view_as_real(data_crop).flip(2).moveaxis(1,-2).flip(2)

    return kspace_crop.numpy()

def main(args):
    # Load the raw kspace:
    # PMoC3D_root = '/media/ssd3/PMoC3D'
    raw_data_path_root = os.path.join(args.PMoC3D_root, 'sourcedata')
    if args.save_path is not None:
        save_path = args.save_path
    else:
        # Default save path
        save_path = os.path.join(args.PMoC3D_root, 'dervatives','cropped_data')

    progress_bar = tqdm(range(1,9), desc='Processing subjects', unit='subject')
    for subject_ind in progress_bar:
        smap_file_name = f'sub-0{subject_ind}/sub-0{subject_ind}_smaps.h5'
        with open(os.path.join(raw_data_path_root, smap_file_name), 'rb') as f:
            smaps = h5py.File(f, 'r')['sens_maps'][()]
        freq_smaps = fft2c_ndim(torch.from_numpy(smaps),3)
        freq_smaps = crop_kspace(freq_smaps.numpy())
        smaps_crop = ifft2c_ndim(torch.from_numpy(freq_smaps),3).cuda(args.gpu)
        binary_background_mask = torch.sum(complex_mul(complex_conj(smaps_crop),smaps_crop),0)[:,:,:,0:1]
        tmp_mask = binary_background_mask.clone()
        tmp_mask[abs(binary_background_mask)<0.001] = float("Inf")
        smaps_crop = (smaps_crop / torch.sqrt(tmp_mask)).cpu().numpy()
        # Save the cropped smaps:
        save_file_name = f'sub-0{subject_ind}/sub-0{subject_ind}_smaps.h5'
        if not os.path.exists(os.path.join(save_path, save_file_name)):
            os.makedirs(os.path.dirname(os.path.join(save_path, save_file_name)), exist_ok=True)
            with h5py.File(os.path.join(save_path, save_file_name), 'w') as f:
                f.create_dataset('sens_maps', data=smaps_crop)

        scan_ind_list = range(5) if subject_ind in [3,5,8] else range(4)
        for scan_ind in scan_ind_list:
            file_name = f'sub-0{subject_ind}/sub-0{subject_ind}_run-0{scan_ind}_kspace.h5'
            with open(os.path.join(raw_data_path_root, file_name), 'rb') as f:
                kspace = h5py.File(f, 'r')['kspace'][()]
                ky_points = h5py.File(f, 'r')['ky_points'][()]
                kz_points = h5py.File(f, 'r')['kz_points'][()]
                mask = h5py.File(f, 'r')['mask'][()]
            # Crop the kspace data:
            kspace_crop = crop_kspace(kspace)
            # Save the cropped kspace data:
            save_file_name = f'sub-0{subject_ind}/sub-0{subject_ind}_run-0{scan_ind}_kspace.h5'
            if not os.path.exists(os.path.join(save_path, save_file_name)):
                os.makedirs(os.path.dirname(os.path.join(save_path, save_file_name)), exist_ok=True)
                with h5py.File(os.path.join(save_path, save_file_name), 'w') as f:
                    f.create_dataset('kspace_crop', data=kspace_crop)
                    f.create_dataset('ky_points', data=ky_points)
                    f.create_dataset('kz_points', data=kz_points)
                    f.create_dataset('mask', data=mask)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Crop PMoC3D dataset')
    parser.add_argument('--gpu', type=int, default=3, help='GPU index to use')
    parser.add_argument('--PMoC3D_root', type=str, default='/media/ssd3/PMoC3D', help='Path to the raw PMoC3D data')
    parser.add_argument('--save_path', type=str, default=None, help='Path to the save directory')
    args = parser.parse_args()
    main(args)

