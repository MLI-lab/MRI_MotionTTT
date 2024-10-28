import torch
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid

from functions.utils.helpers.helpers_math import complex_abs
from functions.utils.helpers.helpers_img_metrics import PSNR_torch
from functions.utils.motion_simulation.motion_forward_backward_models import rotate_translate_3D_complex_img


def get_maxKsp_shot(kspace3D, traj, fix_mot_maxksp_shot, verbose=False):
    '''
    This function goes through all coils and finds the shot that contains the index
    with the maximum k-space entry. An error is raised if the maximum k-space index
    is in multiple shots. If fix_mot_maxksp_shot is True, the shot with the maximum
    k-space index is returned and no gradients are computed for this shot.
    '''
    kspace3D_abs = complex_abs(kspace3D)
    # Log 4x4 center energy
    kspace3D_abs_sumCoil_sumFreq = torch.sum(kspace3D_abs, dim=(0,3), keepdim=False)
    tmp_shape = kspace3D_abs_sumCoil_sumFreq.shape

    if verbose:
        print(torch.where(kspace3D_abs_sumCoil_sumFreq==torch.max(kspace3D_abs_sumCoil_sumFreq)), torch.max(kspace3D_abs_sumCoil_sumFreq))
    center_size = 8
    if tmp_shape[0]%2 == 0:
        ll_x = center_size//2
        uu_x = center_size//2
        center_size_x = center_size
    else:
        ll_x = center_size//2-1
        uu_x = center_size//2
        center_size_x = center_size-1
    if tmp_shape[1]%2 == 0:
        ll_y = center_size//2
        uu_y = center_size//2
        center_size_y = center_size
    else:
        ll_y = center_size//2-1
        uu_y = center_size//2
        center_size_y = center_size-1
    if verbose:
        torch.set_printoptions(linewidth=200)
        print(f"Center {center_size_x}x{center_size_y} k-space energy (summed over coils and freq enc): \n{torch.round(kspace3D_abs_sumCoil_sumFreq[tmp_shape[0]//2-ll_x:tmp_shape[0]//2+uu_x,tmp_shape[1]//2-ll_y:tmp_shape[1]//2+uu_y])}")

    # Get which shots contain the 4x4 center k-space entries
    shot_indices_center = np.zeros((center_size,center_size))
    for ii,i in enumerate(range(tmp_shape[0]//2-ll_x,tmp_shape[0]//2+uu_x)):
        for jj,j in enumerate(range(tmp_shape[1]//2-ll_y,tmp_shape[1]//2+uu_y)):
            for shot in range(len(traj[0])):
                for s in range (len(traj[0][shot])):
                    if i == traj[0][shot][s] and j == traj[1][shot][s]:                    
                        shot_indices_center[ii,jj] = shot
                    
    if verbose:
        logging.info(f"Shots containing the {center_size_x}x{center_size_y} center k-space entries: \n{shot_indices_center}")
    
    # Inspect max k-space index across coils
    shots_with_max_idx = []
    max_indices = []
    for coil in range(kspace3D.shape[0]):
        max_idx = torch.where(kspace3D_abs[coil] == torch.max(kspace3D_abs[coil]))
        max_indices.append((int(max_idx[0][0].cpu().numpy()), int(max_idx[1][0].cpu().numpy())))
        for shot in range(len(traj[0])):
            for s in range (len(traj[0][shot])):
                if max_idx[0][0].cpu().numpy() == traj[0][shot][s] and max_idx[1][0].cpu().numpy() == traj[1][shot][s]:
                    shots_with_max_idx.append(shot)
                    break
    
    unique, counts = np.unique(shots_with_max_idx, return_counts=True)

    if verbose:
        logging.info(f"Shots with max idx across coils: {shots_with_max_idx} at indices {max_indices}")
        logging.info(f"Unique shots with max idx across coils: {unique} (count: {counts}) at unique indices {set(max_indices)}")
        if len(np.unique(shots_with_max_idx)) > 1:
            logging.info("WARNING: Max idx across coils is in separate shots.")

    if fix_mot_maxksp_shot:
        shot_ind_maxksp = unique[np.argmax(counts)]
        logging.info(f"Shot with max k-space energy: {shot_ind_maxksp} (for {counts[np.argmax(counts)]} out of {kspace3D_abs.shape[0]} coils) for which NO gradiets are computed")
    else:
        shot_ind_maxksp = None

    return shot_ind_maxksp



def compute_discretization_error(pred_motion_params, traj, gt_motion_params):
    '''
    Given the current resolution (i.e. the number of k-space lines per motion state) 
    of the predicted motion parameters defined by traj, this function computes the 
    error of the continuous ground truth motion parameters with respect to the ground
    truth motion parameters discretized to the the current resolution.
    '''

    gt_motion_params_discrete = torch.zeros_like(pred_motion_params)
    assert len(traj[0]) == pred_motion_params.shape[0]
    running_ind = 0
    for i in range(len(traj[0])):
        gt_motion_params_discrete[i] = torch.mean(gt_motion_params[running_ind:running_ind+len(traj[0][i])], dim=0)
        running_ind += len(traj[0][i])

    gt_motion_params_discrete_streched, _, _ = expand_mps_to_kspline_resolution(gt_motion_params_discrete, traj, list_of_track_dc_losses=None)
    discretization_error = torch.sum(torch.abs(gt_motion_params_discrete_streched-gt_motion_params))/torch.prod(torch.tensor(gt_motion_params.shape))

    return discretization_error



def DC_loss_thresholding(dc_loss_per_state_norm_per_state, threshold, gt_traj, traj, gt_motion_params, pred_motion_params, masks2D_all_states, masked_corrupted_kspace3D):
    '''
    Input:
        - dc_loss_per_state_norm_per_state: tensor of shape (Ns,) with the normalized DC loss per state
        - threshold: threshold for the DC loss
        - gt_traj: tuple, where gt_traj[0]/gt_traj[1] contains a list of k-space-line-many x/y-coordinates
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
        - gt_motion_params: tensor of shape (number of k-space lines, 6) with the ground truth motion parameters
        - pred_motion_params: tensor of shape (Ns, 6) with the predicted motion parameters
        - masks2D_all_states: tensor of shape (Ns, 1, phase_enc1, phase_enc2, 1, 1)
        - masked_corrupted_kspace3D: tensor of shape (coils, phase_enc1, phase_enc2, freq_enc, 1)
    '''
    if gt_motion_params is not None:
        # # Apply thresholding before expansion (required for masks2D_all_states and masked_corrupted_kspace3D)
        dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state < threshold)[0]
        logging.info(f"Hard DC thresholding applied with threshold {threshold}. Num states before Th: {pred_motion_params.shape[0]} and after Th: {len(dc_th_states_ind)}")

        pred_motion_params_dc = pred_motion_params[dc_th_states_ind]
        Ns = pred_motion_params_dc.shape[0]
        traj_dc= ([traj[0][i] for i in dc_th_states_ind], [traj[1][i] for i in dc_th_states_ind])

        # if gt_motion_params does not have the same number of states as pred_motion_params, we need to 
        # 1. expand gt_motion_params, pred_motion_params, traj and gt_traj to match the number of motion states
        # 2. perform thresholding based on the expanded dc_loss_per_state_norm_per_state
        # 3. Apply thresholding to gt_motion_params, pred_motion_params, and gt_traj
        # 4. Use those to obtain an aligned version of pred_motion_params
        # 5. Reduce the aligned pred_motion_params to the original number of motion states

        # Expand pred_motion_params (required for alignment) and dc_loss_per_state_norm_per_state 
        # (required for thresholding of gt_motion_params) to k-space line resolution

        list_of_track_dc_losses = [torch.from_numpy(dc_loss_per_state_norm_per_state)]
        pred_mp_streched, list_of_track_dc_losses_aligned, reduce_indicator = expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=list_of_track_dc_losses)
        #logging.info(f"Expand pred_motion_params to match k-space line resolution. Num states before expansion: {pred_motion_params.shape[0]} and after expansion: {pred_mp_streched.shape[0]}")
        
        dc_loss_per_state_norm_per_state = list_of_track_dc_losses_aligned[0]
        # # Apply thresholding after extension (required for gt_motion_params and aligned pred_motion_params)
        dc_th_states_ind_extended = np.where(dc_loss_per_state_norm_per_state < threshold)[0]
        #logging.info(f"Hard DC thresholding applied with threshold {threshold}. Num states before Th: {pred_mp_streched.shape[0]} and after Th: {len(dc_th_states_ind_extended)}")

        # Update gt_motion_params, gt_traj according to thresholding
        gt_motion_params = gt_motion_params[dc_th_states_ind_extended]
        gt_traj= ([gt_traj[0][i] for i in dc_th_states_ind_extended], [gt_traj[1][i] for i in dc_th_states_ind_extended])
        
        # Align expanded pred_motion_params to thresholded gt_motion_params
        pred_mp_streched_th = pred_mp_streched[dc_th_states_ind_extended]
        discretization_error = compute_discretization_error(pred_motion_params_dc, traj_dc, gt_motion_params)
        #logging.info(f"L1 loss of motion parameters after DC thresholding: {torch.sum(torch.abs(pred_mp_streched_th-gt_motion_params))/torch.prod(torch.tensor(gt_motion_params.shape))} vs. discretization error after DC thresholding: {discretization_error}")
        pred_mp_streched_th_aligned = motion_alignment(pred_mp_streched_th.cpu(), gt_motion_params.cpu(), r=10, num_points=5001, gpu=None) 
        #logging.info(f"L1 loss of aligned motion parameters after DC thresholding: {torch.sum(torch.abs(pred_mp_streched_th_aligned-gt_motion_params.cpu()))/torch.prod(torch.tensor(gt_motion_params.shape))}")

        # Reduce the aligned version of pred_motion_params to the original number of motion states
        reduce_indicator_th = reduce_indicator[dc_th_states_ind_extended]
        reduce_indicator_th_shifted = torch.zeros_like(reduce_indicator_th)
        reduce_indicator_th_shifted[0] = reduce_indicator_th[0]-1
        reduce_indicator_th_shifted[1:] = reduce_indicator_th[:-1]
        difference = reduce_indicator_th - reduce_indicator_th_shifted
        reduce_indices = torch.where(difference != 0)[0]
        pred_mp_streched_th_aligned_reduced = pred_mp_streched_th_aligned[reduce_indices]
        assert pred_mp_streched_th_aligned_reduced.shape[0] == pred_motion_params_dc.shape[0], "Aligned motion parameters must have the same length as the original motion parameters"

    else:
        discretization_error = None
        # # Apply thresholding
        dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state < threshold)[0]
        #logging.info(f"Hard DC thresholding applied with threshold {threshold}. Num states before Th: {pred_motion_params.shape[0]} and after Th: {len(dc_th_states_ind)}")

        #logging.info(f"Update pred_motion_params, gt_motion_params, traj, masked_corrupted_kspace3D and masks2D_all_states accordingly.")
        pred_motion_params_dc = pred_motion_params[dc_th_states_ind]
        Ns = pred_motion_params_dc.shape[0]

        traj_dc= ([traj[0][i] for i in dc_th_states_ind], [traj[1][i] for i in dc_th_states_ind])


    masks2D_all_states = masks2D_all_states[dc_th_states_ind]

    masked_corrupted_kspace3D_TH = torch.zeros_like(masked_corrupted_kspace3D)
    for i in range(masks2D_all_states.shape[0]):
        masked_corrupted_kspace3D_TH = masked_corrupted_kspace3D_TH + masks2D_all_states[i]*masked_corrupted_kspace3D
    masked_corrupted_kspace3D = masked_corrupted_kspace3D_TH.clone()

    return masked_corrupted_kspace3D, gt_traj, traj_dc, gt_motion_params, pred_motion_params_dc, masks2D_all_states, Ns, dc_th_states_ind, discretization_error

def motion_alignment(mp_pred, mp_gt, r, num_points,gpu):
    '''
    Function is used for align the motion parameters.
    Inputs:
    * mp_pred: estimated motion parameters
    * mp_gt: ground truth motion parameters
    * r: range of the alignment
    * num_points: number of points searched for the alignment
    * gpu: gpu used for the program
    Output: Aligned Motion Predictions
    '''
    base_align = (mp_pred[0]).cpu().numpy()
    if gpu is not None:
        align_final = torch.zeros(6).cuda(gpu)
    else:
        align_final = torch.zeros(6)
    for i in range(6):
        align_set = np.linspace(base_align[i]-r,base_align[i]+r,num_points)
        motion_mae_total = []
        for align in align_set:
            mp_est_align=mp_pred[:,i]-align
            motion_mae_total.append(abs(mp_est_align-mp_gt[:,i]).mean().item())
        align_final[i] = align_set[np.argmin(np.array(motion_mae_total))]
        # print(f'{i+1}/{6} Finished')
    return mp_pred - align_final


def expand_mps_to_kspline_resolution(pred_motion_params, traj, list_of_track_dc_losses=None):
    '''
    This function streches motion parameters to the k-space line resolution.
    '''
    len_pred_traj = len(traj[0])
    pred_mp_aligned = pred_motion_params[0:1,:].repeat(len(traj[0][0]),1)

    # Introduce a 'reduce indicator' to enable reducing the number of motion states to the original number of motion states
    # The reduce indicator has the same length as pred_mp_aligned (i.e. number of k-space lines)
    # Each k-space batch in traj recieves and index 0,...,len_gt_traj-1.
    # For each k-space line in a batch the index is repeated. 
    # Hence, the reduce indicator looks e.g. like [0,0,0,1,1,1,2,2,2,...]
    # This allows to apply thresholding and alignment to the expanded motion parameters and then reduce them to the original number of motion states
    reduce_indicator = torch.zeros(len(traj[0][0]))

    if list_of_track_dc_losses is not None:
        list_of_track_dc_losses_aligned = [list_of_track_dc_losses[i][0:1].repeat(len(traj[0][0])) for i in range(len(list_of_track_dc_losses))]
    else:
        list_of_track_dc_losses_aligned = [None]

    for i in range(len_pred_traj-1):
        pred_mp_aligned = torch.cat((pred_mp_aligned, pred_motion_params[i+1:i+2,:].repeat(len(traj[0][i+1]),1)), dim=0)
        reduce_indicator = torch.cat((reduce_indicator, torch.ones(len(traj[0][i+1]))*(i+1)), dim=0)

        if list_of_track_dc_losses is not None:
            for j in range(len(list_of_track_dc_losses)):
                list_of_track_dc_losses_aligned[j] = torch.cat((list_of_track_dc_losses_aligned[j], list_of_track_dc_losses[j][i+1:i+2].repeat(len(traj[0][i+1]))), dim=0)

    return pred_mp_aligned, list_of_track_dc_losses_aligned, reduce_indicator

def align_3D_volumes(recon_image_fg_1c, target_image_fg_1c, binary_background_mask_2D, gpu):
        # Align reconstructions to target images

        base_align = np.array([0,0,0,0,0,0])
        max_iteration = 25
        r_range = 5

        for i in range(max_iteration):
            r = np.array([1,1,1,1,1,1])*r_range
            num_points = np.array([1,1,1,1,1,1])*3
            max_psnr,mp_align_set = alignment_img_volumes(recon_image_fg_1c.unsqueeze(0),target_image_fg_1c.unsqueeze(0), binary_background_mask_2D.unsqueeze(0).squeeze(-1), base_align, r, num_points, gpu)
            
            if sum(torch.tensor(base_align)-mp_align_set)==0:
                r_range=r_range/2
            else:
                base_align = mp_align_set
                
            #print(f'{i+1}/{max_iteration} Finished.')

        recon_image3D_align = rotate_translate_3D_complex_img(recon_image_fg_1c.unsqueeze(0), mp_align_set, weight_rot=True, gpu=gpu)*binary_background_mask_2D.unsqueeze(0).squeeze(-1)
        return recon_image3D_align[0]

def alignment_img_volumes(recon_image3D, ref_image3D, binary_background_mask3D, base_align, r, num_points, gpu):
        param_grid = {'tx': np.linspace(base_align[0]-r[0],base_align[0]+r[0],num_points[0]), 
                        'ty': np.linspace(base_align[1]-r[1],base_align[1]+r[1],num_points[1]), 
                        'tz': np.linspace(base_align[2]-r[2],base_align[2]+r[2],num_points[2]), 
                        'ra': np.linspace(base_align[3]-r[3],base_align[3]+r[3],num_points[3]), 
                        'rb': np.linspace(base_align[4]-r[4],base_align[4]+r[4],num_points[4]), 
                        'rc': np.linspace(base_align[5]-r[5],base_align[5]+r[5],num_points[5]),
                        }
        num_iteation_total = len(list(ParameterGrid(param_grid)))
        align_total = []
        count = 0
        for para_test in list(ParameterGrid(param_grid)):
            mp_align_set = torch.zeros(6)
            mp_align_set[0] = para_test['tx']
            mp_align_set[1] = para_test['ty']
            mp_align_set[2] = para_test['tz']
            mp_align_set[3] = para_test['ra']
            mp_align_set[4] = para_test['rb']
            mp_align_set[5] = para_test['rc']
            # print(mp_align_set)
            recon_image3D_align = rotate_translate_3D_complex_img(recon_image3D, mp_align_set, weight_rot=True, gpu=gpu)

            if recon_image3D_align.shape[-1]==2:
                align_total.append(PSNR_torch(complex_abs(recon_image3D_align)*binary_background_mask3D, complex_abs(ref_image3D)))
            else:
                align_total.append(PSNR_torch(recon_image3D_align*binary_background_mask3D, ref_image3D))
            count+=1
        max_ind = np.argwhere(np.array(align_total)==max(align_total))
        mp_align_set = torch.zeros(6)
        para_test = list(ParameterGrid(param_grid))[max_ind[0,0]]
        mp_align_set[0] = para_test['tx']
        mp_align_set[1] = para_test['ty']
        mp_align_set[2] = para_test['tz']
        mp_align_set[3] = para_test['ra']
        mp_align_set[4] = para_test['rb']
        mp_align_set[5] = para_test['rc']
        #print(f'The current psnr is {max(align_total)}\n Alignment is: {mp_align_set}')
        return max(align_total),mp_align_set