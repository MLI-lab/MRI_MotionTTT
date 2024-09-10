import os
import h5py
import pickle
import torch
import logging

from functions.helpers.helpers_math import complex_abs, complex_mul, ifft2c_ndim, complex_conj
from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume

def cc359_loader(args):

    ###############   
    # Load k-space, sensitivity maps and mask
    filepath = os.path.join(args.data_drive, args.TTT_example_path)
    filename = filepath.split("/")[-1]

    smap_file = os.path.join(args.data_drive, args.TTT_sensmaps_path, "smaps_"+filename)
    with h5py.File(smap_file, 'r') as hf:
        smaps3D = hf['smaps'][()]
    smaps3D = torch.from_numpy(smaps3D)
    smaps3D_conj = complex_conj(smaps3D)
    binary_background_mask = torch.round(torch.sum(complex_mul(smaps3D_conj,smaps3D),0)[:,:,:,0:1])
    binary_background_mask = binary_background_mask.unsqueeze(0)
    binary_background_mask = binary_background_mask.cuda(args.gpu)

    with h5py.File(filepath, "r") as hf:
        ref_kspace3D = hf["kspace"][()]    
    ref_kspace3D = torch.from_numpy(ref_kspace3D)    

    with open(os.path.join(args.data_drive, args.TTT_mask_path),'rb') as fn:
        mask3D = pickle.load(fn)
        mask3D = torch.tensor(mask3D).unsqueeze(0).unsqueeze(-1) 
        logging.info(f"Using mask from {args.TTT_mask_path}")

    # Compute fully sampled and undersampled image volumes and load to gpu
    ref_img3D_coil = ifft2c_ndim(ref_kspace3D, 3)
    ref_img3D = complex_mul(ref_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

    masked_kspace3D = ref_kspace3D * mask3D
    
    # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
    masked_kspace3D = masked_kspace3D.cuda(args.gpu)
    ref_img3D = complex_abs(ref_img3D.cuda(args.gpu))
    ref_img3D_coil = ref_img3D_coil.cuda(args.gpu)
    smaps3D = smaps3D.cuda(args.gpu)
    smaps3D_conj = smaps3D_conj.cuda(args.gpu)
    mask3D = mask3D.cuda(args.gpu)
    ref_kspace3D = ref_kspace3D.cuda(args.gpu)
    binary_background_mask = binary_background_mask.cuda(args.gpu)

    return ref_img3D, mask3D, masked_kspace3D, smaps3D_conj, ref_kspace3D, ref_img3D_coil, binary_background_mask, smaps3D

def invivo_loader(args, result_path):
     
     ###############   
    # Load k-space, sensitivity maps and mask
    filepath = os.path.join(args.data_drive, args.TTT_example_path)
    motion_corrupted_filename = filepath.split('/')[-1]
    #path_motion_corrupted_scanner_recon = os.path.join(args.data_drive, f"motion_corrupted_stefan/ref_scanner/{motion_corrupted_filename}")

    smap_file = os.path.join(args.data_drive, args.TTT_sensmaps_path)
    #motion_free_filename = smap_file.split('/')[-1]
    #path_motion_free_scanner_recon = os.path.join(args.data_drive, f"motion_corrupted_stefan/ref_scanner/{motion_free_filename}")

    with h5py.File(smap_file, 'r') as hf:
        smaps3D = hf['sens_maps'][()]
        #masked_motion_free_kspace3D = hf['kspace_crop'][()]
        masked_motion_free_kspace3D = None
    smaps3D = torch.from_numpy(smaps3D)
    smaps3D_conj = complex_conj(smaps3D)
    binary_background_mask = torch.round(torch.sum(complex_mul(smaps3D_conj,smaps3D),0)[:,:,:,0:1])
    binary_background_mask = binary_background_mask.unsqueeze(0)
    binary_background_mask = binary_background_mask.cuda(args.gpu)

    with h5py.File(filepath, "r") as hf:
        masked_corrupted_kspace3D = hf["kspace_crop"][()] 
        mask3D = hf['mask'][()]
        ky_points = hf['ky_points'][()]
        kz_points = hf['kz_points'][()]

    #with h5py.File(path_motion_free_scanner_recon, "r") as hf:
    #    motion_free_scanner_recon_img3D = hf["reference"][()]

    #with h5py.File(path_motion_corrupted_scanner_recon, "r") as hf:
    #    motion_corrupted_scanner_recon_img3D = hf["reference"][()]

    masked_corrupted_kspace3D = torch.from_numpy(masked_corrupted_kspace3D) 
    #masked_motion_free_kspace3D = torch.from_numpy(masked_motion_free_kspace3D)
    mask3D = torch.tensor(mask3D)

    # # Save some slices from the scanner reconstruction of the volume to which
    # we want to apply TTT and the volume from which we use the sensitivity maps
    #h, w, d = motion_corrupted_scanner_recon_img3D.shape
    #list_of_slices = [(2, d//2), (1, w//2), (0, h//2), (0, h//2-1), (0, h//2-2), (0, h//2+1), (0, h//2+2)]
    #save_slice_images_from_volume(motion_corrupted_scanner_recon_img3D, list_of_slices, result_path, "motion_corrupted_scanner_recon", axis_names = ["axial","coronal","saggital"], dir_name="scanner_recons")
    #save_slice_images_from_volume(motion_free_scanner_recon_img3D, list_of_slices, result_path, "motion_free_scanner_recon", axis_names = ["axial","coronal","saggital"], dir_name="scanner_recons")

    # All 3D img or kspace volumes must be of shape (coils, X, Y, Z, 2) or (X, Y, Z, 2)
    smaps3D = smaps3D.cuda(args.gpu)
    smaps3D_conj = smaps3D_conj.cuda(args.gpu)
    mask3D = mask3D.cuda(args.gpu)
    binary_background_mask = binary_background_mask.cuda(args.gpu)
    masked_corrupted_kspace3D = masked_corrupted_kspace3D.cuda(args.gpu) #1.3GB
    #masked_motion_free_kspace3D = masked_motion_free_kspace3D.cuda(args.gpu)

    ###############
    # Generate sampling trajectory
    nl = len(ky_points)//args.Ns
    ky_points = ky_points+masked_corrupted_kspace3D.shape[1]//2
    kz_points = kz_points+masked_corrupted_kspace3D.shape[2]//2

    traj = ([ky_points[i*nl:(i+1)*nl] for i in range(args.Ns)], [kz_points[i*nl:(i+1)*nl] for i in range(args.Ns)])
    
    ###############

    return masked_corrupted_kspace3D, masked_motion_free_kspace3D, mask3D, smaps3D_conj, traj, binary_background_mask, smaps3D