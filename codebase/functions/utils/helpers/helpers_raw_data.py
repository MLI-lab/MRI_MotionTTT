import numpy as np
import pandas as pd
import torch
import mat73
import h5py

from functions.utils.helpers.helpers_math import complex_mul, ifft2c_ndim, fft2c_ndim, complex_conj

def load_data(matfilename,listfilename,crop_freq_enc_to_size = None):
    '''
    Load the kspace data from the ".mat" file
    Load the sampling sequence and mask from the ".list" file
    Input: 
        - matfilename: ".mat" file name containing kspace information
        - listfilename: ".list" file name containing mask information
        - crop_freq_enc_to_size: Number of slices in freq encoding direction after crop
    Output:
        - full kspace
        - Croped kspace
        - ky_points: sampling sequence on the y axis
        - kz_points: sampling sequence on the z axis
        - mask: mask for the sampling
    '''
    # Load .mat file
    mat = mat73.loadmat(matfilename)
    #mat = h5py.File(matfilename, 'r')
    scanner_recon = mat['reference']
    #scanner_recon = mat['reference'][()]
    if scanner_recon is None:
        print('No reference image found in the mat file')
    # Remove Padding on the k-space
    Nx = int(mat['encoding_pars']['KxRange'][1] - mat['encoding_pars']['KxRange'][0] + 1)
    Ny = int(2 * (mat['encoding_pars']['KyRange'][1]+1))
    Nz = int(2 * mat['encoding_pars']['KzRange'][1])
    kspace_sorted_shape = mat['kspace_sorted'].shape
    kspace = mat['kspace_sorted'][(kspace_sorted_shape[0]-Nx)//2:(kspace_sorted_shape[0]+Nx)//2,
                                (kspace_sorted_shape[1]-Ny)//2:(kspace_sorted_shape[1]+Ny)//2,
                                (kspace_sorted_shape[2]-Nz)//2+1:(kspace_sorted_shape[2]+Nz)//2+1,
                                :]
    # kspace = mat['kspace_sorted'][()][:,(kspace_sorted_shape[1]-Nx)//2:(kspace_sorted_shape[1]+Nx)//2,
    #                             (kspace_sorted_shape[2]-Ny)//2:(kspace_sorted_shape[2]+Ny)//2,
    #                             (kspace_sorted_shape[3]-Nz)//2+1:(kspace_sorted_shape[3]+Nz)//2+1,]
    # kspace = kspace['real']+1j*kspace['imag']
    # fftshift on the image domain:
    kspace = torch.view_as_real(torch.from_numpy(kspace).moveaxis(-1,0))
    #kspace = torch.view_as_real(torch.from_numpy(kspace))
    # Get sampling seqience and mask
    ky_points, kz_points = get_sampling_sequence(listfilename)
    mask = get_mask(kspace.shape,ky_points,kz_points)
    # Shift the data into center (corresponds to shifting the image by half the axis length on second and third axis)
    kspace_shift = kspace_ifftshift(kspace,mask)

    if crop_freq_enc_to_size is not None:
        # Crop the data on the frequency encoding direction:
        data_cpx = torch.view_as_complex(kspace_shift)
        data_cpx = torch.fft.ifftshift(data_cpx, dim=(1))
        data_cpx = torch.fft.ifftn(data_cpx, dim=(1), norm="ortho")
        data_cpx = torch.fft.fftshift(data_cpx, dim=(1))
        # Crop the data
        data_crop = data_cpx[:,(Nx-crop_freq_enc_to_size)//2:(Nx+crop_freq_enc_to_size)//2,...]
        # 1D back transform:
        data_crop = torch.fft.fftshift(data_crop, dim=(1))
        data_crop = torch.fft.fftn(data_crop, dim=(1), norm="ortho")
        data_crop = torch.fft.ifftshift(data_crop, dim=(1))
        kspace_crop = torch.view_as_real(data_crop)
    else:
        kspace_crop = kspace_shift

    #mat.close()

    return (kspace_shift.flip(2).moveaxis(1,-2).flip(2), 
           kspace_crop.flip(2).moveaxis(1,-2).flip(2),
           (-1*ky_points-1), 
           (-1*kz_points-1),
           mask.flip(2).moveaxis(1,-2).flip(2),
           mat,
           scanner_recon)

def get_sampling_sequence(listfilename):
    key_names = ['typ','mix','dyn','card','echo','loca','chan','extr1','extr2','ky','kz','n.a.','aver','sign','rf','grad','enc','rtop','rr','size','offset','rphase','mphase','pda','pda_f','t_dyn','codsiz','chgrp','format','norm_fac','kylab','kzlab']
    data = pd.read_csv(listfilename, sep=' ',skiprows=5, header=None, names=key_names, dtype={'ID':str}, skipinitialspace=True)

    # # # Inspect data object
    # # Ste the display options to show all columns
    # pd.set_option('display.max_columns', None)
    # # Display the first 5 rows of the data
    # print(data.head())
    # # Display the shape of the table, should be (about 150k, 32)
    # print(data.shape)
    # print(data['typ'].unique()) # should be [3, 5, 1]
    # print(data['chan'].unique()) 

    measurements = data[data["typ"] == 1]

    # print(measurements.head())
    # print(measurements.shape)

    channels_numbers = np.array(measurements["chan"].unique())
    ky_points = np.array(measurements["ky"], dtype=int)[::len(channels_numbers)]
    kz_points =np.array(measurements["kz"], dtype=int)[::len(channels_numbers)]

    # print(ky_points.shape)
    # print(kz_points.shape)

    return (ky_points,
            kz_points)
    
def get_mask(kspace_size,ky_points,kz_points):
    mask = torch.zeros(kspace_size[2:-1]).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
    ky_points = ky_points+kspace_size[2]//2
    kz_points = kz_points+kspace_size[3]//2
    mask[:,:,ky_points,kz_points,:] = 1
    
    return mask

def load_sensmap(matfile_name, crop_freq_enc_to_size=None, mat=None):

    if mat is None:
        mat = mat73.loadmat(matfile_name)
    #mat = h5py.File(matfile_name, 'r')
    #sens_maps = mat['smaps'][()]
    #sens_maps = sens_maps['real']+1j*sens_maps['imag']
    sens_maps = mat['smaps']
    sens_maps = torch.from_numpy(sens_maps)
    freq_sens_maps = torch.view_as_complex(fft2c_ndim(torch.view_as_real(sens_maps.moveaxis(-1,0)),3))
    #freq_sens_maps = torch.view_as_complex(fft2c_ndim(torch.view_as_real(sens_maps),3))

    Nx = int(mat['encoding_pars']['KxRange'][1] - mat['encoding_pars']['KxRange'][0] + 1)
    Ny = int(2 * (mat['encoding_pars']['KyRange'][1]+1))
    Nz = int(2 * mat['encoding_pars']['KzRange'][1])
    freq_sens_maps_shape = freq_sens_maps.shape
    freq_sens_maps = freq_sens_maps[:,
                                (freq_sens_maps_shape[1]-Nx)//2:(freq_sens_maps_shape[1]+Nx)//2,
                                (freq_sens_maps_shape[2]-Ny)//2:(freq_sens_maps_shape[2]+Ny)//2,
                                (freq_sens_maps_shape[3]-Nz)//2+1:(freq_sens_maps_shape[3]+Nz)//2+1]
    freq_sens_maps = torch.view_as_real(torch.tensor(freq_sens_maps.clone().detach()))

    sens_maps = ifft2c_ndim(freq_sens_maps.flip(2).moveaxis(1,-2).flip(2), 3)

    sens_maps_conj = complex_conj(sens_maps)
    binary_background_mask = torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,:,0:1]

    positions_nonzero = binary_background_mask.squeeze()!=0
    tmp_mask = binary_background_mask.clone()
    tmp_mask[abs(binary_background_mask)<0.1] = float("Inf")
    sens_maps_norm = sens_maps / torch.sqrt(tmp_mask)


    if crop_freq_enc_to_size is not None:
        # Crop the data on the frequency encoding direction:
        data_cpx = torch.view_as_complex(freq_sens_maps)
        data_cpx = torch.fft.ifftshift(data_cpx, dim=(1))
        data_cpx = torch.fft.ifftn(data_cpx, dim=(1), norm="ortho")
        data_cpx = torch.fft.fftshift(data_cpx, dim=(1))
        # Crop the data
        data_crop = data_cpx[:,(Nx-crop_freq_enc_to_size)//2:(Nx+crop_freq_enc_to_size)//2,...]
        # 1D back transform:
        data_crop = torch.fft.fftshift(data_crop, dim=(1))
        data_crop = torch.fft.fftn(data_crop, dim=(1), norm="ortho")
        data_crop = torch.fft.ifftshift(data_crop, dim=(1))
        freq_sens_maps_cropped = torch.view_as_real(data_crop).flip(2).moveaxis(1,-2).flip(2)

        sens_maps = ifft2c_ndim(freq_sens_maps_cropped, 3)

        sens_maps_conj = complex_conj(sens_maps)
        binary_background_mask = torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,:,0:1]

        positions_nonzero = binary_background_mask.squeeze()!=0
        tmp_mask = binary_background_mask.clone()
        tmp_mask[abs(binary_background_mask)<0.1] = float("Inf")
        sens_maps_norm_cropped = sens_maps / torch.sqrt(tmp_mask)

    #mat.close()

    return sens_maps_norm, sens_maps_norm_cropped
    
def kspace_ifftshift(kspace,mask):
    '''
    Shift the image on the kspace to keep the mask unchanged
    Input:
        - kspace: kspace of the sampled data
        - mask: undersampling mask
        - shift_dim: dimensions require shift
    Output:
        - kspace_shift
    '''
    x_dim, y_dim, z_dim = kspace.shape[1], kspace.shape[2], kspace.shape[3]

    idx = torch.from_numpy(np.arange(x_dim)-x_dim//2)#.cuda(args.gpu)
    idy = torch.from_numpy(np.arange(y_dim)-y_dim//2)#.cuda(args.gpu)
    idz = torch.from_numpy(np.arange(z_dim)-z_dim//2)#.cuda(args.gpu)

    pshift = idx[:,None,None].repeat(1,len(idy),len(idz))*0.
    pshift += idy[None,:,None].repeat(len(idx),1,len(idz))*0.5
    pshift += idz[None,None:].repeat(len(idx),len(idy),1)*0.5
    pshift = torch.view_as_real(torch.exp(1j*2*np.pi*pshift))#.unsqueeze(0)#.cuda(args.gpu)

    kspace_shift = complex_mul(pshift,kspace)

    return kspace_shift*mask
    


