import matplotlib.pyplot as plt
import torchvision
import io
import os
import torch
import numpy as np
import logging
import k3d
from typing import Optional, Sequence, Tuple, Union
from functions.helpers.helpers_math import complex_abs, center_crop

def print_gpu_memory_usage(self, step_index):

        current_memory = torch.cuda.memory_allocated(self.args.gpu)  # device_id is the ID of your GPU
        peak_memory = torch.cuda.max_memory_allocated(self.args.gpu)
        current_memory_reserved = torch.cuda.memory_reserved(self.args.gpu)
        peak_memory_reserved = torch.cuda.max_memory_reserved(self.args.gpu)
        logging.info(f"{step_index} Current GPU memory usage: {current_memory / 1024**3} GB")
        logging.info(f"{step_index} Peak GPU memory usage: {peak_memory / 1024**3} GB")
        logging.info(f"{step_index} Current GPU memory reserved: {current_memory_reserved / 1024**3} GB")
        logging.info(f"{step_index} Peak GPU memory reserved: {peak_memory_reserved / 1024**3} GB")

def save_modelTTT_curves(name, save_path, modelTTT_meters_per_example, final_result_dict_modelTTT):

    best_step_selfval_before_earlyStopped = final_result_dict_modelTTT["best_step_selfval_before_earlyStopped_modelTTT"]
    psnr_at_best_epoch_selfval_beforeES = final_result_dict_modelTTT["earlyStopped_PSNR_ref_vs_recon_modelTTT"]

    psnr_at_best_psnr = final_result_dict_modelTTT["best_PSNR_ref_vs_recon_modelTTT"]
    epoch_at_best_psnr = final_result_dict_modelTTT["best_step_PSNR_ref_vs_recon_modelTTT"]

    figure = plt.figure(figsize=(12,8))
    ax1 = figure.add_subplot(121)
    ax1.plot(modelTTT_meters_per_example['modelTTT_fit_loss'].epochs,modelTTT_meters_per_example['modelTTT_fit_loss'].val,color='b',label='self fit')
    ax1.plot(modelTTT_meters_per_example['modelTTT_selfval_loss'].epochs,modelTTT_meters_per_example['modelTTT_selfval_loss'].val,color='g',label='self val')
    ax1.axvline(x=best_step_selfval_before_earlyStopped, color='k', linestyle='--',label=f"best selfval before ES psnr={psnr_at_best_epoch_selfval_beforeES:.2f}")
    ax1.axvline(x=epoch_at_best_psnr, color='r', linestyle='--',label=f"best psnr={psnr_at_best_psnr:.2f}")
    ax1.axvline(x=final_result_dict_modelTTT["earlyStopped_at_step"], color='m', linestyle='--',label=f"early stopping (ES)")
    # make ylabel
    ax1.set_ylabel('L1 loss')
    ax1.set_xlabel('TTT epoch')
    ax1.legend()

    ax3 = figure.add_subplot(122)
    ax3.plot(modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].epochs,modelTTT_meters_per_example["modelTTT_PSNR_recon_ref"].val,color='b',label='psnr sense')
    ax3.axvline(x=best_step_selfval_before_earlyStopped, color='k', linestyle='--',label=f"best selfval before ES psnr={psnr_at_best_epoch_selfval_beforeES:.2f}")
    ax3.axvline(x=epoch_at_best_psnr, color='r', linestyle='--',label=f"best psnr={psnr_at_best_psnr:.2f}")
    ax3.axvline(x=final_result_dict_modelTTT["earlyStopped_at_step"], color='m', linestyle='--',label=f"early stopping (ES)")
    # make ylabel
    ax3.set_ylabel('psnr')
    ax3.set_xlabel('TTT epoch')
    ax3.legend()

    #plt.title(f"TTT early stop epoch was {TTT_per_sample_track['early_stopp_epoch']}")

    plt.savefig(os.path.join(save_path, name + ".png"))
    plt.close(figure)


def save_figure_sensmaps(name, save_path, sens_maps, sens_maps_learned):
    num_coils = sens_maps.shape[1]
    # pick two random coils at once
    coils = np.random.randint(0,num_coils,1)

    for coil in coils:
        s_est_abs = complex_abs(sens_maps[0:1,coil:coil+1,:,:,:])
        s_learn_abs = complex_abs(sens_maps_learned[0:1,coil:coil+1,:,:,:])
        abs_min = min(s_est_abs.min(),s_learn_abs.min())
        s_est_abs = s_est_abs - abs_min
        s_learn_abs = s_learn_abs - abs_min
        abs_max = max(s_est_abs.max(),s_learn_abs.max())
        s_est_abs = s_est_abs / abs_max
        s_learn_abs = s_learn_abs / abs_max


        s_est_re = sens_maps[0:1,coil:coil+1,:,:,0]
        s_est_im = sens_maps[0:1,coil:coil+1,:,:,1]

        s_learn_re = sens_maps_learned[0:1,coil:coil+1,:,:,0]
        s_learn_im = sens_maps_learned[0:1,coil:coil+1,:,:,1]

        re_im_min = min(s_est_re.min(),s_learn_re.min(),s_est_im.min(),s_learn_im.min())
        s_est_re = s_est_re - re_im_min
        s_learn_re = s_learn_re - re_im_min
        s_est_im = s_est_im - re_im_min
        s_learn_im = s_learn_im - re_im_min

        re_im_max = max(s_est_re.max(),s_learn_re.max(),s_est_im.max(),s_learn_im.max())
        s_est_re = s_est_re / re_im_max
        s_learn_re = s_learn_re / re_im_max
        s_est_im = s_est_im / re_im_max
        s_learn_im = s_learn_im / re_im_max

        image = torch.cat([s_est_abs, s_est_re, s_est_im, s_learn_abs, s_learn_re, s_learn_im], dim=0)
    
        image = torchvision.utils.make_grid(image, nrow=3, normalize=False, pad_value=1.0)

        header = f"Coil {coil} | Row 1/2 = espirit/learned | columns 1/2/3 = abs/re/im"
        figure = get_figure(image.cpu().numpy(),figsize=(7,13),title=header)
        plt.savefig(os.path.join(save_path, name + f"_c{coil}.pdf"))
        plt.close(figure)
    pass


def add_img_to_tensorboard(writer, epoch, name, input_img, target, output, val_ssim_fct):

    input_img = complex_abs(input_img)
    if target.shape[-1] == 2:
        target = complex_abs(target)
    if output.shape[-1] == 2:
        output = complex_abs(output)

    input_img = input_img.unsqueeze(0)
    output = output.unsqueeze(0)
    target = target.unsqueeze(0)

    # Normalize output to mean and std of target
    #target, output = normalize_to_given_mean_std(target, output)               
    ssim_loss = 1-val_ssim_fct(output, target, data_range=target.max().unsqueeze(0))

    error = torch.abs(target - output)
    input_img = input_img - input_img.min() 
    input_img = input_img / input_img.max()
    output = output - output.min() 
    output = output / output.max()
    target = target - target.min()
    target = target / target.max()
    error = error - error.min() 
    error = error / error.max()
    image = torch.cat([input_img, target, output, error], dim=0)
    image = torchvision.utils.make_grid(image, nrow=1, normalize=False, pad_value=1.0)
    #if crop:
    figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"ssim={ssim_loss.item():.6f}")
    #else:
    #    figure = get_figure(image.cpu().numpy(),figsize=(3,20),title=f"ssim={ssim_loss.item():.6f}")
    writer.add_image(name, plot_to_image(figure), epoch)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    #img = buf.getvalue()#.to(torch.uint8)
    #img = torch.import_ir_module_from_buffer(img,dtype=torch.uint8)
    frameTensor = torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8), device='cpu')
    image = torchvision.io.decode_png(frameTensor)
    # Add the batch dimension
    #image = torch.unsqueeze(image, 0)
    return image

def get_figure(image,figsize,title):
    """Return a matplotlib figure of a given image."""
    if len(image.shape) != 3:
        raise ValueError("Image dimensions not suitable for logging to tensorboard.")
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.rollaxis(image,0,3)
    # Create a figure to contain the plot.
    if figsize:
        figure = plt.figure(figsize=figsize)
    else:
        figure = plt.figure()
    # Start next subplot.
    plt.subplot(1, 1, 1, title=title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    figure.tight_layout()

    return figure

def save_figure(
    x,
    figname: str,
    save_path: str,
    save_npz: bool = False,
    crop_size:tuple = None,):
    """"
    x must have dimension height,width if it is np array
    If it has last dim 2 it must be a torch tensor of shape 1,height,width,2
    """
    if x.shape[-1] == 2:
        x = complex_abs(x)
        if crop_size is not None:
            x = center_crop(x, crop_size)
        x = x.squeeze().detach().cpu().numpy()
                
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(x,'gray')
    ax.axis('off')
    #ax.set_title(title,fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, figname + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if save_npz:
        np.savez(os.path.join(save_path, figname + ".npz"), 
                figname=x)
        
def save_figure_original_resolution(x, save_path, figname, vmin=None, vmax=None):
    fig = plt.figure(figsize=(x.shape[1]/100, x.shape[0]/100), dpi=100)
    plt.imshow(x, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(os.path.join(save_path, figname + ".png"), dpi=100, pad_inches=0)
    plt.close(fig)


def save_slice_images_from_volume(volume, list_of_slices, save_path, volume_name, axis_names = ["X","Y","Z"], kspace=False, coil_index=None, dir_name="slice_images"):
    '''
    Save some specified slices of 3D volume as images to file.
    Input:
        - volume: Torch tensor of shape (X,Y,Z,2), (num_coils,X,Y,Z,2), (X,Y,Z) or (num_coils,X,Y,Z).
        - list_of_slices: List of tuples, where each tuple (which axis to slice, which slice to save)
            where "which axis to slice must be in [0,1,2]
        - save_path: At this path this function creates slice_images folder and saves the images there.
        - axis_names: List of strings with the names of the axes. Default is ["X","Y","Z"].
        - kspace: If True, the volume is assumed to be k-space data and the images are saved in log scale.
        - coil_index: Index of coil to save. Required if volume has coil dimension.
        - dir_name: Name of the folder where the images will be saved. Default is "slice_images".
    '''

    assert len(volume.shape) in [3,4,5], "Volume must have shape (X,Y,Z,2), (num_coils,X,Y,Z,2), (X,Y,Z) or (num_coils,X,Y,Z)"

    if len(volume.shape) == 5 or (len(volume.shape) == 4 and volume.shape[-1] != 2):
        assert coil_index is not None, "Coil index must be specified for coil volume."

    # Create folder for saving images
    save_path = os.path.join(save_path, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if coil_index is not None:
        volume = volume[coil_index]

    if volume.shape[-1] == 2:
        volume = complex_abs(volume)

    if kspace:
        volume = torch.log(volume +1e-9)

    if list_of_slices is None:
        list_of_slices = [(0, volume.shape[0]//2), (1, volume.shape[1]//2), (2, volume.shape[2]//2)]
        
    for slice in list_of_slices:
        axis = slice[0]
        slice_index = slice[1]

        if axis == 0:
            slice_image = volume[slice_index,:,:]
        elif axis == 1:
            slice_image = volume[:,slice_index,:]
        elif axis == 2:
            slice_image = volume[:,:,slice_index]

        if kspace:
            save_figure_original_resolution(slice_image, save_path, f"{volume_name}_ksp_{axis_names[axis]}_{slice_index}")
        else:
            save_figure_original_resolution(slice_image, save_path, f"{volume_name}_img_{axis_names[axis]}_{slice_index}")



def volume_to_k3d_html(volume : np.array, filter_upper_percentile : float = 95.0, 
                       filter_lower_percentile : float = 5.0, bounds_shape : Optional[Tuple[int, int, int]] = None, 
                       volume_name = "k3d_vol", save_path = None, dir_name = "k3d_volumes"):
    plot = k3d.plot(lighting=0.0)
    
    #logging.info(“Creating k3d volume...this might take some time”)
    print("Creating k3d volume...this might take some time")
    if bounds_shape is None:
        bounds = None
    else:
        bounds = [0, bounds_shape[2], 0, bounds_shape[1], 0, bounds_shape[0]]
    plt_volume = k3d.volume(volume,
            color_range=[
                np.percentile(volume, filter_lower_percentile),
                np.percentile(volume, filter_upper_percentile)
            ],
            interpolation=False,
            compression_level=9,
            alpha_coef=100,
            gradient_step=0.00005,
            bounds = bounds, #[0, volume.shape[2], 0, volume.shape[1], 0, volume.shape[0]],
            #color_map=k3d.paraview_color_maps.X_Ray)
            color_map=k3d.paraview_color_maps.Grayscale)
    plot += plt_volume
    plot.display()
    print("Finished creating k3d volume")

    save_path = os.path.join(save_path, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename_3D = os.path.join(save_path, f"{volume_name}.html")
    with open(filename_3D, 'w') as f:
        f.write(plot.get_snapshot())
    #return plot.get_snapshot()

def save_masks_from_motion_sampling_trajectories(traj, mask2d, save_path, num_states_to_save=None, dir_name="motion_sampling_traj", save_figures = False):

    # Create folder for saving images
    save_path = os.path.join(save_path, dir_name)
    if save_figures and not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if num_states_to_save is None:
        num_states_to_save = len(traj[0])+1

    if save_figures:
        save_figure_original_resolution(mask2d, save_path, "mask2d_phase_slice_encoding")

    # masks2D_all_states has shape (num_states_to_save, 1 (coil dim), 
    # mask2d.shape[0], mask2d.shape[1], 1 (broadcast freq dim), 1 (comlex dim))
    masks2D_all_states = np.zeros((num_states_to_save-1, 1, mask2d.shape[0], mask2d.shape[1], 1, 1))

    sanity_check_mask = np.zeros_like(mask2d)
    for i in range(num_states_to_save-1):
        mask_traj = np.zeros_like(mask2d)
        mask_traj[traj[0][i], traj[1][i]] = 1
        if save_figures:
            save_figure_original_resolution(mask_traj, save_path, f"mask_traj_{i}")
        masks2D_all_states[i, 0, :, :, 0, 0] = mask_traj

        assert np.sum(sanity_check_mask*mask_traj) == 0, "Some mask trajectories overlap"
        sanity_check_mask += mask_traj
    logging.info(f"Num lines with motion state in traj: {np.sum(sanity_check_mask)}")
    logging.info(f"Num lines in mask: {np.sum(mask2d)}")

    return masks2D_all_states