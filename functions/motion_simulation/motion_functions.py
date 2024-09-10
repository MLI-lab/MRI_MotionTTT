import torch
import torch.nn.functional as F
import numpy as np
import torchkbnufft as tkbn
import time
import logging
import scipy.signal as signal
import os
import pickle

from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, normalize_separate_over_ch
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj, chunks

#from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex

from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html
from functions.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories

#import functions.motion_simulation.kbnufft as nufftkb
import functions.motion_simulation.kbnufft as nufftkb_forward
import functions.motion_simulation.kbnufft_2 as nufftkb_adjoint

def get_maxKsp_shot(kspace3D, traj, fix_mot_maxksp_shot):
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
        # # Apply peak or hard thresholding before expansion (required for masks2D_all_states and masked_corrupted_kspace3D)
        if threshold == "peak":
            dc_th_states_ind_to_exclude = signal.find_peaks(dc_loss_per_state_norm_per_state)[0]
            dc_th_states_ind = np.setdiff1d(np.arange(0,dc_loss_per_state_norm_per_state.shape[0]), dc_th_states_ind_to_exclude)
            logging.info(f"Peak DC thresholding applied. Num states before Th: {pred_motion_params.shape[0]} and after Th: {len(dc_th_states_ind)}")
        else:
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
        logging.info(f"Expand pred_motion_params to match k-space line resolution. Num states before expansion: {pred_motion_params.shape[0]} and after expansion: {pred_mp_streched.shape[0]}")
        
        dc_loss_per_state_norm_per_state = list_of_track_dc_losses_aligned[0]
        # # Apply peak or hard thresholding after extension (required for gt_motion_params and aligned pred_motion_params)
        if threshold == "peak":
            dc_th_states_ind_to_exclude = signal.find_peaks(dc_loss_per_state_norm_per_state)[0]
            dc_th_states_ind_extended = np.setdiff1d(np.arange(0,dc_loss_per_state_norm_per_state.shape[0]), dc_th_states_ind_to_exclude)
            logging.info(f"Peak DC thresholding applied. Num states before Th: {pred_mp_streched.shape[0]} and after Th: {len(dc_th_states_ind)}")
        else:
            dc_th_states_ind_extended = np.where(dc_loss_per_state_norm_per_state < threshold)[0]
            logging.info(f"Hard DC thresholding applied with threshold {threshold}. Num states before Th: {pred_mp_streched.shape[0]} and after Th: {len(dc_th_states_ind_extended)}")

        # Update gt_motion_params, gt_traj according to thresholding
        gt_motion_params = gt_motion_params[dc_th_states_ind_extended]
        gt_traj= ([gt_traj[0][i] for i in dc_th_states_ind_extended], [gt_traj[1][i] for i in dc_th_states_ind_extended])
        
        # Align expanded pred_motion_params to thresholded gt_motion_params
        pred_mp_streched_th = pred_mp_streched[dc_th_states_ind_extended]
        discretization_error = compute_discretization_error(pred_motion_params_dc, traj_dc, gt_motion_params)
        logging.info(f"L1 loss of motion parameters after DC thresholding: {torch.sum(torch.abs(pred_mp_streched_th-gt_motion_params))/torch.prod(torch.tensor(gt_motion_params.shape))} vs. discretization error after DC thresholding: {discretization_error}")
        pred_mp_streched_th_aligned = motion_alignment(pred_mp_streched_th.cpu(), gt_motion_params.cpu(), r=10, num_points=5001, gpu=None) 
        logging.info(f"L1 loss of aligned motion parameters after DC thresholding: {torch.sum(torch.abs(pred_mp_streched_th_aligned-gt_motion_params.cpu()))/torch.prod(torch.tensor(gt_motion_params.shape))}")

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
        # # Apply peak or hard thresholding
        if threshold == "peak":
            dc_th_states_ind_to_exclude = signal.find_peaks(dc_loss_per_state_norm_per_state)[0]
            dc_th_states_ind = np.setdiff1d(np.arange(0,dc_loss_per_state_norm_per_state.shape[0]), dc_th_states_ind_to_exclude)
            logging.info(f"Peak DC thresholding applied. Num states before Th: {pred_motion_params.shape[0]} and after Th: {len(dc_th_states_ind)}")
        else:
            dc_th_states_ind = np.where(dc_loss_per_state_norm_per_state < threshold)[0]
            logging.info(f"Hard DC thresholding applied with threshold {threshold}. Num states before Th: {pred_motion_params.shape[0]} and after Th: {len(dc_th_states_ind)}")

        logging.info(f"Update pred_motion_params, gt_motion_params, traj, masked_corrupted_kspace3D and masks2D_all_states accordingly.")
        pred_motion_params_dc = pred_motion_params[dc_th_states_ind]
        Ns = pred_motion_params_dc.shape[0]

        traj_dc= ([traj[0][i] for i in dc_th_states_ind], [traj[1][i] for i in dc_th_states_ind])


    masks2D_all_states = masks2D_all_states[dc_th_states_ind]

    masked_corrupted_kspace3D_TH = torch.zeros_like(masked_corrupted_kspace3D)
    for i in range(masks2D_all_states.shape[0]):
        masked_corrupted_kspace3D_TH = masked_corrupted_kspace3D_TH + masks2D_all_states[i]*masked_corrupted_kspace3D
    masked_corrupted_kspace3D = masked_corrupted_kspace3D_TH.clone()

    return masked_corrupted_kspace3D, gt_traj, traj_dc, gt_motion_params, pred_motion_params_dc, masks2D_all_states, Ns, dc_th_states_ind, discretization_error

def sim_motion_get_gt_motion_traj(args, traj):

    if args.motionTraj_simMot == "uniform_interShot_event_model":
        logging.info(f"Generate inter-shot random motion parameters with seed {args.random_motion_seed}, motion states {args.Ns}, number of shots {args.TTT_num_shots}, max translation/rotation {args.max_trans}/{args.max_rot}, num motion events {args.num_motion_events}")
        gt_motion_params = gen_rand_mot_params_interShot(args.Ns, args.max_trans, args.max_rot, args.random_motion_seed, args.num_motion_events, args.TTT_num_shots)
        #traj_updated = traj
        # Ectending this to k-space resolution makes L1 recon with known motion way too slow but
        # we need it to compute loss on motion parameters
        gt_motion_params,_,_ = expand_mps_to_kspline_resolution(gt_motion_params, traj, list_of_track_dc_losses=None)

        traj_updated = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])
        for i in torch.arange(1,args.Ns):
            # For each shot expand the traj to per line resolution
            traj_updated[0].extend([np.array([k]) for k in traj[0][i]])
            traj_updated[1].extend([np.array([k]) for k in traj[1][i]])

        intraShot_event_inds = None
    elif args.motionTraj_simMot == "uniform_intraShot_event_model":
        logging.info(f"Generate intra-shot random motion parameters with seed {args.random_motion_seed}, motion states {args.Ns}, number of shots {args.TTT_num_shots}, max translation/rotation {args.max_trans}/{args.max_rot}, num motion events {args.num_motion_events}, num intraShot events {args.num_intraShot_events}")
        gt_motion_params, traj_updated, intraShot_event_inds = gen_rand_mot_params_intraShot(args.Ns, args.max_trans, args.max_rot, 
                                                                                                          args.random_motion_seed, args.num_motion_events, 
                                                                                                          args.num_intraShot_events, args.TTT_num_shots, traj)
        #gt_mp_ksp_reso = gt_motion_params
        logging.info(f"Number of motion states in traj without intra-shot motion {len(traj[0])} and with intra-shot motion {len(traj_updated[0])}")
    else:
        raise ValueError(f"motionTraj_simMot {args.motionTraj_simMot} not implemented.")

    return gt_motion_params.cuda(args.gpu), traj_updated, intraShot_event_inds#, gt_mp_ksp_reso.cuda(args.gpu)

def sim_motion_get_traj(args, mask3D):

    if args.TTT_sampTraj_simMot == "interleaved_cartesian":
        logging.info(f"Generate interleaved cartesian sampling trajectory with center_in_first_state {args.center_in_first_state}")
        traj, _ = generate_interleaved_cartesian_trajectory(args.Ns, mask3D, args, center_in_first_state=args.center_in_first_state)

    elif args.TTT_sampTraj_simMot == "interleaved_cartesian_fixNumShots":
        logging.info(f"Generate interleaved cartesian sampling trajectory with center_in_first_state {args.center_in_first_state} and fixNumShots {args.TTT_num_shots}")
        traj, _ = generate_interleaved_cartesian_trajectory_fixNumShots(args.Ns, mask3D, args, center_in_first_state=args.center_in_first_state, num_shots=args.TTT_num_shots)

    elif args.TTT_sampTraj_simMot == "interleaved_cartesian_Ns500":
        logging.info(f"Generate interleaved cartesian sampling trajectory with center_in_first_state {args.center_in_first_state} and fixNumShots {args.TTT_num_shots}")
        logging.info("First interleaved sampling is applied with 500 states which are then combined to TTT_num_shots many shots")
    
        traj_tmp, _ = generate_interleaved_cartesian_trajectory_fixNumShots(500, mask3D, args, center_in_first_state=args.center_in_first_state, num_shots=500)
        traj = ([],[])
        assert 500 % args.TTT_num_shots == 0, "Number of states must be divisible by number of shots."
        states_per_shot = 500 // args.TTT_num_shots
        for i in range(args.TTT_num_shots):
            traj[0].append(np.concatenate([traj_tmp[0][j] for j in range(i*states_per_shot,(i+1)*states_per_shot)]))
            traj[1].append(np.concatenate([traj_tmp[1][j] for j in range(i*states_per_shot,(i+1)*states_per_shot)]))

    elif args.TTT_sampTraj_simMot == "random_cartesian":
        logging.info(f"Generate random cartesian sampling trajectory with center_in_first_state {args.center_in_first_state}")
        traj, _ = generate_random_cartesian_trajectory(args.Ns, mask3D, args, center_in_first_state=args.center_in_first_state, seed=args.random_sampTraj_seed)

    elif args.TTT_sampTraj_simMot == "deterministic_cartesian":
        logging.info(f"Load deterministic cartesian sampling trajectory from {args.sampling_order_path}")
        traj, _ = generate_deterministic_cartesian_trajectory(args.Ns, mask3D, args, path_to_traj=args.sampling_order_path)

    elif args.TTT_sampTraj_simMot == "linear_cartesian":
        logging.info(f"Generate linear cartesian sampling trajectory.")
        traj, _ = generate_linear_cartesian_trajectory(args.Ns, mask3D, args)
    else:
        raise ValueError(f"TTT_sampTraj_simMot {args.TTT_sampTraj_simMot} not implemented.")

    return traj


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

def gen_rand_mot_params_interShot(Ns, max_trans, max_rot, seed, num_events, TTT_num_shots):
    '''
    Draw num_events many out of Ns motion states that receive a unique patient position.
    The remaining motion states get the patient position of the last event.
    One position is defined by 3 translations and 3 rotations, which are drawn uniformly
    from [-max_rot/-max_trans, max_rot/max_trans]
    Input:
        - Ns: number of motion states
        - max_trans: maximum translation in pixels
        - max_rot: maximum rotation in degrees
        - seed: random seed
        - num_events: number of motion states with unique patient positions
        - TTT_num_shots: number of shots
    Output:
        - motion_params: tensor of shape (Ns, 6) with the motion parameters    
        - motion_params_shots_states_map: array with one entry per shot containing the number of motion states in this shot
    '''
    assert Ns == TTT_num_shots, "Number of motion states must match number of shots for inter-shot motion simulation."
    motion_params = torch.zeros(Ns, 6)
    motion_params_events = torch.zeros(num_events, 6)
    torch.manual_seed(seed)
    motion_params_events[:,0:3] = torch.rand([num_events,3]) * 2 * max_trans - max_trans
    motion_params_events[:,3:6] = torch.rand([num_events,3]) * 2 * max_rot - max_rot

    #motion_params_shots_states_map = torch.zeros(TTT_num_shots)

    # pick random motion states for the events and sort them
    # remove the zero entry from event_states (we assume not motion during the first shot)
    event_states = torch.randperm(Ns)
    event_states = event_states[event_states != 0]
    event_states = event_states[:num_events]
    
    event_states = torch.sort(event_states)[0]
    for i in range(len(event_states)):
        if i == len(event_states)-1:
            motion_params[event_states[i]:,:] = motion_params_events[i:i+1,:]
        else:
            motion_params[event_states[i]:event_states[i+1],:] = motion_params_events[i:i+1,:]

    #for i in range(TTT_num_shots):
    #    motion_params_shots_states_map[i] = 1


    return motion_params

def gen_rand_mot_params_intraShot(Ns, max_trans, max_rot, seed, num_events, num_intraShot_events, TTT_num_shots, traj):
    '''
    Draw num_events many out of Ns motion states that receive a unique patient position.
    The remaining motion states get the patient position of the last event.
    One position is defined by 3 translations and 3 rotations, which are drawn uniformly
    from [-max_rot/-max_trans, max_rot/max_trans].
    num_intraShot_events-many events receive a motion state per line within this shot.
    Those states linearly interpolate following and preceeding motion parameters.
    Input:
        - Ns: number of motion states
        - max_trans: maximum translation in pixels
        - max_rot: maximum rotation in degrees
        - seed: random seed
        - num_events: number of shots with unique patient positions
        - num_intraShot_events: number of motion shots with one motion state per line
        - TTT_num_shots: number of shots. Must be equal to Ns
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    Output:
        - motion_params: tensor of shape (number of k-space lines, 6) with ground truth the motion parameters    
        - traj_updated: tuple, where traj_updated[0]/traj_updated[1] contains a list of number-of-k-space-lines-many x/y-coordinates
        - events_intrashot: list of shot indices with intra-shot motion
    '''
    assert Ns == TTT_num_shots, "Number of motion states must match number of shots for intra-shot motion simulation."
    assert num_intraShot_events <= num_events, "Number of intra-shot events must be smaller or equal to the number of events."

    # Generate inter-shot motion parameters (one set of 6 parameters for each shot following the event model)
    motion_params_inter = torch.zeros(Ns, 6)
    motion_params_events = torch.zeros(num_events, 6)
    #torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    motion_params_events[:,0:3] = torch.rand([num_events,3], generator=gen) * 2 * max_trans - max_trans
    motion_params_events[:,3:6] = torch.rand([num_events,3], generator=gen) * 2 * max_rot - max_rot

    # pick random motion shots for where the motion events take place and sort them (exclude first shot)
    event_states_shuffled = torch.arange(1,Ns)
    event_states_shuffled = event_states_shuffled[torch.randperm(len(event_states_shuffled), generator=gen)][:num_events]
    event_states = torch.sort(event_states_shuffled)[0]

    # Assign generated motion parameters to the shots following the event model
    for i in range(len(event_states)):
        if i == len(event_states)-1:
            motion_params_inter[event_states[i]:,:] = motion_params_events[i:i+1,:]
        else:
            motion_params_inter[event_states[i]:event_states[i+1],:] = motion_params_events[i:i+1,:]

    # from the event shots randomly pick the ones that are intra-shot motion events
    events_intrashot = event_states_shuffled[:num_intraShot_events]
    logging.info(f"Shot indices with intra-shot motion: {events_intrashot}")
    
    # In this model the first shot has constant all-zero motion parameters
    motion_params = torch.zeros(len(traj[0][0]), 6)
    traj_updated = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])

    for i in torch.arange(1,TTT_num_shots):
        # For each shot expand the traj to per line resolution
        traj_updated[0].extend([np.array([k]) for k in traj[0][i]])
        traj_updated[1].extend([np.array([k]) for k in traj[1][i]])
        
        if i in events_intrashot:
            if i == Ns-1:
                # There is a intra-shot event in the last shot. 
                # Concatenate random motion state to motion_params_inter to compute ending points of the intra-shot motion
                pass
            # Design intra-shot motion
            num_lines = len(traj[0][i])

            # Decide whether motion has already started during break
            if torch.rand(1, generator=gen) < 0.3:
                # motion starts withing this shot, somewhere in the first third of the shot
                num_starting_lines_const = int(torch.rand(1, generator=gen) / 3 * num_lines)
            else:
                # motion started before
                num_starting_lines_const = 0

            # Decide whether motion ends during that shot
            if torch.rand(1, generator=gen) < 0.3:
                # motion ends withing this shot, somewhere in the last third of the shot
                num_ending_lines_const = int(torch.rand(1, generator=gen) / 3 * num_lines)
            else:
                # motion continues
                num_ending_lines_const = 0

            num_lines_motion = num_lines - num_starting_lines_const - num_ending_lines_const
            motion_params_intra = torch.zeros(num_lines, 6)

            for j in range(6):
                
                # if motion starts or ends during the shot add motion parameters of the previous or next shot accordingly
                motion_params_intra[:num_starting_lines_const,j] = motion_params_inter[i-1,j]
                motion_params_intra[num_lines-num_ending_lines_const:,j] = motion_params_inter[i,j]

                shot_gap = torch.abs(motion_params_inter[i-1,j] - motion_params_inter[i,j]).item()

                if num_starting_lines_const == 0:
                    # generate offset at start of shot with random fraction of the shot gap and random sign
                    offset_fraction = torch.rand(1, generator=gen).item() / 5
                    offset_sign = torch.sign(torch.rand(1, generator=gen) - 0.5).item()
                    starting_point = motion_params_inter[i-1,j].item() + offset_sign * offset_fraction * shot_gap
                else:
                    starting_point = motion_params_inter[i-1,j].item()

                if num_ending_lines_const == 0:
                    # generate offset at end of shot with random fraction of the shot gap and random sign
                    offset_fraction = torch.rand(1, generator=gen).item() / 5
                    offset_sign = torch.sign(torch.rand(1, generator=gen) - 0.5).item()
                    ending_point = motion_params_inter[i,j].item() - offset_sign * offset_fraction * shot_gap
                else:
                    ending_point = motion_params_inter[i,j].item()

                # determine the motion parameters for the lines with motion
                # determine number of peaks
                tmp = torch.rand(1, generator=gen)
                motion_ratio = num_lines_motion / num_lines
                if tmp < motion_ratio/3:
                    if tmp < motion_ratio/6:
                        # two peaks
                        if torch.rand(1, generator=gen) < 0.5:
                            # first overshoot towards the direction of the next motion state
                            ending_point_peak_1 = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                            ending_point_peak_2 = ending_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        else:
                            # first have peak in the opposite direction of the next motion state
                            ending_point_peak_1 = starting_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                            ending_point_peak_2 = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/5).item()

                        motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion//3,j] = torch.linspace(starting_point, ending_point_peak_1, num_lines_motion//3)
                        motion_params_intra[num_starting_lines_const+num_lines_motion//3:num_starting_lines_const+2*(num_lines_motion//3),j] = torch.linspace(ending_point_peak_1, ending_point_peak_2, num_lines_motion//3)
                        motion_params_intra[num_starting_lines_const+2*(num_lines_motion//3):num_starting_lines_const+num_lines_motion,j] = torch.linspace(ending_point_peak_2, ending_point, num_lines_motion-(2*(num_lines_motion//3)))
                    else:
                        # one peak
                        if torch.rand(1, generator=gen) < 0.5:
                            # overshoot towards the direction of the next motion state
                            ending_point_peak = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        else:
                            # have peak in the opposite direction of the next motion state
                            ending_point_peak = starting_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion//2,j] = torch.linspace(starting_point, ending_point_peak, num_lines_motion//2)
                        motion_params_intra[num_starting_lines_const+num_lines_motion//2:num_starting_lines_const+num_lines_motion,j] = torch.linspace(ending_point_peak, ending_point, num_lines_motion-num_lines_motion//2)
                else:
                    # zero peaks
                    motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion,j] = torch.linspace(starting_point, ending_point, num_lines_motion)

            motion_params = torch.cat((motion_params, motion_params_intra), dim=0)
        else:
            # Repeat the motion parameters of this shot to obtain per line resolution
            motion_params = torch.cat((motion_params, motion_params_inter[i:i+1,:].repeat(len(traj[0][i]),1)), dim=0)
            #motion_params = torch.cat((motion_params, motion_params_inter[i:i+1,:]), dim=0)
            #traj_updated = (traj_updated[0] + [traj[0][i]], traj_updated[1] + [traj[1][i]])

    return motion_params, traj_updated, events_intrashot

def generate_random_cartesian_trajectory(Ns, mask3D, args=None, save_path=None, center_in_first_state=True, seed=0):
    '''
    Given a 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1) the 
    acquired k-space lines are specified by the phase_enc1 and phase_enc2 plane.
    The acquired lines are ordered randomly and assigned to Ns-many batches.
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 
    if center_in_first_state:
        mask2D_center = np.zeros_like(mask2D)
        mask2D_center[mask2D.shape[0]//2-1:mask2D.shape[0]//2+2,mask2D.shape[1]//2-1:mask2D.shape[1]//2+2] = 1
        mask2D_no_center = mask2D - mask2D_center
    
        # assign 3x3 center lines to the same motion state
        recordedx_center, recordedy_center = np.where(mask2D_center==1)
        recordedx, recordedy = np.where(mask2D_no_center==1)
    else:
        recordedx, recordedy = np.where(mask2D==1)

    # shuffle recordedx and recordedy in the same way
    np.random.seed(seed)
    np.random.shuffle(recordedx)
    np.random.seed(seed)
    np.random.shuffle(recordedy)

    if center_in_first_state:
        # attach the center lines to the trajectory of the first motion state
        recordedx = np.concatenate((recordedx_center, recordedx))
        recordedy = np.concatenate((recordedy_center, recordedy))

    traj = (list(chunks(recordedx,Ns)), list(chunks(recordedy,Ns)))

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

def generate_deterministic_cartesian_trajectory(Ns, mask3D, args=None, save_path=None, path_to_traj=None):
    '''
    Load a sampling trajectory from file and chunk it into Ns-many batches.
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 

    with open(os.path.join(args.data_drive, path_to_traj),'rb') as fn:
        order = pickle.load(fn)
    recordedx = order[0][0]
    recordedy = order[1][0]

    traj = (list(chunks(recordedx,Ns)), list(chunks(recordedy,Ns)))

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

def generate_interleaved_cartesian_trajectory(Ns, mask3D, args=None, save_path=None, center_in_first_state=True):
    '''
    Given a 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1) the 
    acquired k-space lines are specified by the phase_enc1 and phase_enc2 plane.
    The acquired lines are assgined to the motion states in an interleaved fashion.
    Hence, if Ns=10 every 10th line is assigned to the same motion state.
    Further, the center 3x3 lines are assigned to the first motion state.
    Input:
        - Ns: number of motion states
        - mask3D: 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1)
        - args: arguments of the experiment
        - save_path: path to save the masks
        - center_in_first_state: if True, the center 3x3 lines are assigned to the first motion state        
    Output:
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 
    if center_in_first_state:
        mask2D_center = np.zeros_like(mask2D)
        mask2D_center[mask2D.shape[0]//2-1:mask2D.shape[0]//2+2,mask2D.shape[1]//2-1:mask2D.shape[1]//2+2] = 1
        mask2D_no_center = mask2D - mask2D_center
    
        # assign 3x3 center lines to the same motion state
        recordedx_center, recordedy_center = np.where(mask2D_center==1)
        recordedx, recordedy = np.where(mask2D_no_center==1)
    else:
        recordedx, recordedy = np.where(mask2D==1)


    recordedx_no_center = recordedx[0:len(recordedx):Ns]
    recordedy_no_center = recordedy[0:len(recordedy):Ns]
    for i in range(1,Ns):
        recordedx_no_center = np.concatenate((recordedx_no_center, recordedx[i:len(recordedx):Ns]))
        recordedy_no_center = np.concatenate((recordedy_no_center, recordedy[i:len(recordedy):Ns]))

    if center_in_first_state:
        recordedx = np.concatenate((recordedx_center, recordedx_no_center))
        recordedy = np.concatenate((recordedy_center, recordedy_no_center))


    traj = (list(chunks(recordedx,Ns)), list(chunks(recordedy,Ns)))
    #traj = ([recordedx[i:len(recordedx):Ns] for i in range(Ns)], [recordedy[i:len(recordedy):Ns] for i in range(Ns)])
    
    # if center_in_first_state:
    #     # attach the center lines to the trajectory of the first motion state
    #     traj[0][0] = np.concatenate((traj[0][0], recordedx_center))
    #     traj[1][0] = np.concatenate((traj[1][0], recordedy_center))

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

def generate_interleaved_cartesian_trajectory_fixNumShots(Ns, mask3D, args=None, save_path=None, center_in_first_state=True, num_shots=50):
    '''
    Given a 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1) the 
    acquired k-space lines are specified by shots in an interleaved fashion.
    Hence, if num_shots=10 every 10th line is assigned to the same shot.
    Further, the center 3x3 lines are assigned to the first shot.
    If e.g. the number of motion states Ns is twice the number of shots, the first and second
    half of each shot is assigned to individual motion states.
    Input:
        - Ns: number of motion states
        - mask3D: 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1)
        - args: arguments of the experiment
        - num_shots: number of shots
        - save_path: path to save the masks
        - center_in_first_state: if True, the center 3x3 lines are assigned to the first shot
    Output:
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 
    if center_in_first_state:
        mask2D_center = np.zeros_like(mask2D)
        mask2D_center[mask2D.shape[0]//2-1:mask2D.shape[0]//2+2,mask2D.shape[1]//2-1:mask2D.shape[1]//2+2] = 1
        mask2D_no_center = mask2D - mask2D_center
    
        # assign 3x3 center lines to the same motion state
        recordedx_center, recordedy_center = np.where(mask2D_center==1)
        recordedx, recordedy = np.where(mask2D_no_center==1)
    else:
        recordedx, recordedy = np.where(mask2D==1)

    traj = ([recordedx[i:len(recordedx):num_shots] for i in range(num_shots)], [recordedy[i:len(recordedy):num_shots] for i in range(num_shots)])
    
    if center_in_first_state:
        # attach the center lines to the trajectory of the first motion state
        traj[0][0] = np.concatenate((traj[0][0], recordedx_center))
        traj[1][0] = np.concatenate((traj[1][0], recordedy_center))

    if Ns > num_shots:
        traj_final = ([],[])
        for i in range(num_shots):
            traj_final[0].extend(list(chunks(traj[0][i],Ns//num_shots)))
            traj_final[1].extend(list(chunks(traj[1][i],Ns//num_shots)))

        traj = traj_final

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

def generate_linear_cartesian_trajectory(Ns, mask3D, args=None, save_path=None, dir='y'):
    '''
    Given a 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1) the 
    acquired k-space lines are specified by the phase_enc1 and phase_enc2 plane.
    The acquired lines are assgined to the motion states in an interleaved fashion.
    Hence, if Ns=10 every 10th line is assigned to the same motion state.
    Further, the center 3x3 lines are assigned to the first motion state.
    Input:
        - Ns: number of motion states
        - mask3D: 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1)
        - args: arguments of the experiment
        - save_path: path to save the masks
        - dir: direction of the sampling trajectory       
    Output:
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 
    if dir=='x':
        mask2d_traj = mask2D.transpose()
    else:
        mask2d_traj = mask2D
        
    mask_coord = []
    mask_length = np.zeros(mask2d_traj.shape[0])
    for i in range(mask2d_traj.shape[0]):
        mask_coord.append(np.where(mask2d_traj[i]==1)[0])
        mask_length[i] = len(np.where(mask2d_traj[i]==1)[0])


    score = np.zeros(len(mask_length))
    current_index = np.zeros(len(mask_length)).astype(int)
    x_coord = []
    y_coord = []
    # For the first loop:
    for x in range(mask2d_traj.shape[0]):
                score[x]+=1/mask_length[x]
                if len(mask_coord[x])==0:
                    continue
                x_coord.append(x)
                y_coord.append(mask_coord[x][0])
                current_index[x] = current_index[x] + 1

    while sum(current_index-mask_length)!=0:
        min_index = np.where(score==min(score))[0]
        for x in min_index:
            if len(mask_coord[x])==0:
                print(1)
                continue
            x_coord.append(x)
            y_coord.append(mask_coord[x][current_index[x]])
            current_index[x] = current_index[x] + 1
            score[x]+=1/mask_length[x]

    # Ns = 52
    #nl = len(y_coord)//Ns

    if dir=='x':
        #traj = ([y_coord[i*nl:(i+1)*nl] for i in range(Ns)], [x_coord[i*nl:(i+1)*nl] for i in range(Ns)])
        traj = (list(chunks(y_coord,Ns)), list(chunks(x_coord,Ns)))
    else:
        #traj = ([x_coord[i*nl:(i+1)*nl] for i in range(Ns)], [y_coord[i*nl:(i+1)*nl] for i in range(Ns)])
        traj = (list(chunks(x_coord,Ns)), list(chunks(y_coord,Ns)))

    # For each pair of x/y-coordinates in traj[0][i]/traj[1][i] sort the x/y-coordinates according to x coordinates in traj[0][i] in ascending order
    for i in range(Ns):
        traj[0][i], traj[1][i] = zip(*sorted(zip(traj[0][i],traj[1][i])))

        

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

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



def unet_forward_all_axes(net, masked_corrected_img3D, rec_id, ax_id):
    '''
    Forward map of the UNet
    Input:
    * net: the trained net for motion-free data reconstruction
    * masked_corrected_img3D: 3D ZF motion-corrected image torch.Tensor (x,y,z,2)
    * sens_maps: 3d sensitivity maps
    * rec_id: the slices that we need to reconsturct
    '''
    assert len(masked_corrected_img3D.shape) == 4, "Input image must have shape (x,y,z,2)"
    assert ax_id in [0,1,2], "ax_id must be 0, 1 or 2"
    assert masked_corrected_img3D.shape[-1] == 2, "Input image must be complex valued"  
    # Move axial dim to the first dim (batch dim)
    masked_corrected_img3D = torch.moveaxis(masked_corrected_img3D,ax_id,0)    
    inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(masked_corrected_img3D,-1,1), eps=1e-11)
    
    # Reconstruct the NN with no grad:
    with torch.no_grad():
        image_recon = net(inputs_img_full_2c_norm)
    # Reconstruct image using grad for defined slices
    if rec_id is not None:
        image_recon[rec_id] = net(inputs_img_full_2c_norm[rec_id])
    
    image_recon = image_recon * std + mean   
    image_recon = torch.moveaxis(image_recon,1,-1)    # (batch, x, y, ch)
    image_recon = torch.moveaxis(image_recon,0,ax_id)    
    return image_recon


def motion_correction_NUFFT(kspace3D, mp, traj, weight_rot, args, do_dcomp=True, num_iters_dcomp=3, grad_translate=True, grad_rotate=True, states_with_grad=None, shot_ind_maxksp=None, max_coil_size=None):
    '''
    Given a 3D k-space this function uses the adjoint NUFFT to compute the off-grid
    k-space values for the acquired lines specidfied in traj and with respect to the
    motion parameters mp. 
    Input:
        - kspace3D: 3D tensor of shape (coils,x,y,z,2)
        - mp: motion parameters a tensor of shape (Ns, 6) with Ns the number of motion states
        and 6 the number of motion parameters (tx,ty,tz,alpha,beta,gamma). translations are in pixels
        and rotations in degrees.
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
        specifying which k-space lines were acquired under which motion state.
        - weight_rot: Boolean, if True, the rotation matrix is weighted to account
        for the aspect ratio of the image
        - args: arguments of the experiment
        - do_dcomp: Boolean, if True, density compensation is applied
        - num_iters_dcomp: number of iterations for the density compensation
        - grad_translate: Boolean, if True, the translation parameters are differentiable
        - grad_rotate: Boolean, if True, the rotation parameters are differentiable
        - states_with_grad: list of motion states that are differentiable
        - shot_ind_maxksp: a single index for which no gradients are computed
        - max_coil_size: maximum number of coils that are processed at once
    Output:
        - img3D: 3D tensor of shape (coils,x,y,z,2).    
    '''

    assert kspace3D.shape[-1] == 2, "Input k-space must be complex valued"
    if mp is None:
        mp = torch.zeros(len(traj[0]),6).cuda(args.gpu)
        #logging.info("No motion parameters provided to motion_correction_NUFFT. Set motion parameters to 0.")
    else:
        assert mp.shape[0] == len(traj[0]), "Number of motion states must match number of trajectory states"
        
    assert len(kspace3D.shape) == 5, "Input k-space must have shape (coils,x,y,z,2)"

    #Ns = mp.shape[0]+1
    Ns = len(traj[0])
    x_dim, y_dim, z_dim = kspace3D.shape[1], kspace3D.shape[2], kspace3D.shape[3]
    w1 = x_dim/y_dim if weight_rot else 1
    w2 = x_dim/z_dim if weight_rot else 1
    w3 = y_dim/x_dim if weight_rot else 1
    w4 = y_dim/z_dim if weight_rot else 1
    w5 = z_dim/x_dim if weight_rot else 1
    w6 = z_dim/y_dim if weight_rot else 1
    IDx = traj[0]
    IDy = traj[1]

    idx = torch.from_numpy(np.arange(x_dim)-x_dim//2).cuda(args.gpu)
    idy = torch.from_numpy(np.arange(y_dim)-y_dim//2).cuda(args.gpu)
    idz = torch.from_numpy(np.arange(z_dim)-z_dim//2).cuda(args.gpu)

    grid_x, grid_y, grid_z = torch.meshgrid(idx,idy,idz, indexing='ij')
    coord = torch.stack((grid_x,grid_y,grid_z),dim=0).type(torch.FloatTensor).cuda(args.gpu)

    for s in range(Ns):

        if states_with_grad is not None:
            if s in states_with_grad:
                grad_translate_tmp = grad_translate
                grad_rotate_tmp = grad_rotate
            else:
                grad_translate_tmp = False
                grad_rotate_tmp = False
        else:
            grad_translate_tmp = grad_translate
            grad_rotate_tmp = grad_rotate

        if shot_ind_maxksp is not None:
            if s == shot_ind_maxksp:
                grad_translate_tmp = False
                grad_rotate_tmp = False

        idx_s = IDx[s]
        idy_s = IDy[s]
        a = mp[s,3]/180*np.pi if grad_rotate_tmp else mp[s,3].detach()/180*np.pi
        b = mp[s,4]/180*np.pi if grad_rotate_tmp else mp[s,4].detach()/180*np.pi
        g = mp[s,5]/180*np.pi if grad_rotate_tmp else mp[s,5].detach()/180*np.pi
        
        transx = mp[s,0] if grad_translate_tmp else mp[s,0].detach()
        transy = mp[s,1] if grad_translate_tmp else mp[s,1].detach()
        transz = mp[s,2] if grad_translate_tmp else mp[s,2].detach()

        pshift = idx[:,None,None].repeat(1,len(idy),len(idz))*transx/x_dim 
        pshift += idy[None,:,None].repeat(len(idx),1,len(idz))*transy/y_dim
        pshift += idz[None,None:].repeat(len(idx),len(idy),1)*transz/z_dim
        pshift = torch.view_as_real(torch.exp(1j*2*np.pi*pshift)).cuda(args.gpu)

        pshift_real = pshift[idx_s,idy_s,:,0] * kspace3D[:,idx_s,idy_s,:,0] - pshift[idx_s,idy_s,:,1] * kspace3D[:,idx_s,idy_s,:,1]
        pshift_imag = pshift[idx_s,idy_s,:,0] * kspace3D[:,idx_s,idy_s,:,1] + pshift[idx_s,idy_s,:,1] * kspace3D[:,idx_s,idy_s,:,0]   

        pshift = pshift_real + 1j*pshift_imag
        if s==0:
            ksp_sampled = pshift.reshape(kspace3D.shape[0],-1)
        else:
            ksp_sampled = torch.cat([ksp_sampled,pshift.reshape(kspace3D.shape[0],-1)],dim=1)

        trans = torch.zeros(3,3).cuda(args.gpu)
        trans[0,0] = torch.cos(a) * torch.cos(b)
        trans[0,1] = w1*(torch.cos(a) * torch.sin(b) * torch.sin(g) - torch.sin(a) * torch.cos(g))
        trans[0,2] = w2*( torch.cos(a) * torch.sin(b) * torch.cos(g) + torch.sin(a) * torch.sin(g))
        trans[1,0] = w3*(torch.sin(a) * torch.cos(b))
        trans[1,1] = torch.sin(a) * torch.sin(b) * torch.sin(g) + torch.cos(a) * torch.cos(g)
        trans[1,2] = w4*(torch.sin(a) * torch.sin(b) * torch.cos(g) - torch.cos(a) * torch.sin(g))
        trans[2,0] = -w5*(torch.sin(b))
        trans[2,1] = w6*(torch.cos(b) * torch.sin(g))
        trans[2,2] = torch.cos(b) * torch.cos(g)

        if s==0:
            coord_rot = coord[:,idx_s,idy_s,:].reshape(3,-1).type(torch.FloatTensor)
            rot_coord_sampled = trans@coord_rot.cuda(args.gpu)
        else:
            coord_rot = coord[:,idx_s,idy_s,:].reshape(3,-1).type(torch.FloatTensor)
            rot_coord_sampled = torch.cat([rot_coord_sampled,trans@coord_rot.cuda(args.gpu)],dim=1)

    rot_coord_sampled[0] = rot_coord_sampled[0]/x_dim*2*torch.pi
    rot_coord_sampled[1] = rot_coord_sampled[1]/y_dim*2*torch.pi
    rot_coord_sampled[2] = rot_coord_sampled[2]/z_dim*2*torch.pi


    nufftkb_adjoint.nufft.set_dims(len(rot_coord_sampled[0]), (kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]), 'cuda:'+str(args.gpu), Nb=3, norm=args.nufft_norm)
    nufftkb_adjoint.nufft.precompute(rot_coord_sampled.moveaxis(0,1))
    #nufftkb.nufft.set_dims(len(rot_coord_sampled[0]), (kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]), 'cuda:'+str(args.gpu), Nb=3, norm=None)
    #nufftkb.nufft.precompute(rot_coord_sampled.moveaxis(0,1))

    if do_dcomp:
        #start = time.time()
        dcomp = tkbn.calc_density_compensation_function(ktraj=rot_coord_sampled.detach(), 
                                                        im_size=(kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]),
                                                        num_iterations = num_iters_dcomp)
        #print(f"Time for dcomp: {time.time()-start} with {num_iters_dcomp} iterations")
    else:
        dcomp=None

    if do_dcomp:
        if max_coil_size is not None:
            coil_list = list(range(0,ksp_sampled.shape[0]))
            coil_list_batches = [coil_list[i:i+max_coil_size] for i in range(0, len(coil_list), max_coil_size)]

            for jj,coil_batch in enumerate(coil_list_batches):
                if jj==0:
                    img3D = nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled[coil_batch]*dcomp[0]))
                else:
                    img3D = torch.cat([img3D, nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled[coil_batch]*dcomp[0]))],dim=0)
        else:
            img3D = nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled*dcomp[0]))
    else:
        if max_coil_size is not None:
            coil_list = list(range(0,ksp_sampled.shape[0]))
            coil_list_batches = [coil_list[i:i+max_coil_size] for i in range(0, len(coil_list), max_coil_size)]

            for jj,coil_batch in enumerate(coil_list_batches):
                if jj==0:
                    img3D = nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled[coil_batch]))
                else:
                    img3D = torch.cat([img3D, nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled[coil_batch]))],dim=0)
        else:
            img3D = nufftkb_adjoint.adjoint(rot_coord_sampled.moveaxis(0,1), torch.view_as_real(ksp_sampled))
        
        
    #if args.nufft_norm is None:
    eng_ratio = torch.sqrt(torch.sum(abs(img3D)**2)/torch.sum(abs(ksp_sampled)**2))
    if eng_ratio.requires_grad:
        eng_ratio = eng_ratio.detach()
    #else:
    #    eng_ratio = 1.0

    img3D = img3D/eng_ratio

    ksp_sampled = None
    rot_coord_sampled = None
    coord = None
    pshift = None
    grid_x = None
    grid_y = None
    grid_z = None
    pshift_real = None
    pshift_imag = None
    dcomp = None
    kspace3D = None

    return img3D#, eng_ratio

def motion_corruption_NUFFT(kspace3D, image3D_coil, mp, traj, weight_rot, args, grad_translate=True, grad_rotate=True, states_with_grad=None, shot_ind_maxksp=None, max_coil_size=None):
    '''
    Given a fully sampled 3D k-space this function uses the NUFFT to compute for each 
    k-space line defined in traj the off-grid k-space values for the corresponding 
    motion state mp. The obtained values are placed on the coordinates specified by traj
    in the Cartesin corrupted k-space.
    Input:
        - kspace3D: 3D tensor of shape (coils,x,y,z,2)
        - mp: motion parameters a tensor of shape (Ns, 6) with Ns the number of motion states
        and 6 the number of motion parameters (tx,ty,tz,alpha,beta,gamma). translations are in pixels
        and rotations in degrees.
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
        specifying which k-space lines were acquired under which motion state.
        - weight_rot: Boolean, if True, the rotation matrix is weighted to account
        for the aspect ratio of the image
        - args: arguments of the experiment
    Output:
        - corrupted_kspace3D: 3D tensor of shape (coils,x,y,z,2).    
    '''

    assert kspace3D.shape[-1] == 2, "Input k-space must be complex valued"
    assert mp.shape[0] == len(traj[0]), "Number of motion states must match number of trajectory states"
    assert len(kspace3D.shape) == 5, "Input k-space must have shape (coils,x,y,z,2)"

    Ns = len(traj[0])
    x_dim, y_dim, z_dim = kspace3D.shape[1], kspace3D.shape[2], kspace3D.shape[3]
    w1 = x_dim/y_dim if weight_rot else 1
    w2 = x_dim/z_dim if weight_rot else 1
    w3 = y_dim/x_dim if weight_rot else 1
    w4 = y_dim/z_dim if weight_rot else 1
    w5 = z_dim/x_dim if weight_rot else 1
    w6 = z_dim/y_dim if weight_rot else 1
    IDx = traj[0]
    IDy = traj[1]
    
    idx = torch.from_numpy(np.arange(x_dim)-x_dim//2).cuda(args.gpu)
    idy = torch.from_numpy(np.arange(y_dim)-y_dim//2).cuda(args.gpu)
    idz = torch.from_numpy(np.arange(z_dim)-z_dim//2).cuda(args.gpu)
    
    grid_x, grid_y, grid_z = torch.meshgrid(idx,idy,idz, indexing='ij')
    coord = torch.stack((grid_x,grid_y,grid_z),dim=0).type(torch.FloatTensor).cuda(args.gpu)
    
    # Step 1: Rotate the data
    for s in range(Ns):
        idx_s = IDx[s]
        idy_s = IDy[s]

        if states_with_grad is not None:
            if s in states_with_grad:
                grad_translate_tmp = grad_translate
                grad_rotate_tmp = grad_rotate
            else:
                grad_translate_tmp = False
                grad_rotate_tmp = False
        else:
            grad_translate_tmp = grad_translate
            grad_rotate_tmp = grad_rotate

        if shot_ind_maxksp is not None:
            if s == shot_ind_maxksp:
                grad_translate_tmp = False
                grad_rotate_tmp = False

        a = -1*mp[s,3]/180*np.pi if grad_rotate_tmp else -1*mp[s,3].detach()/180*np.pi
        b = -1*mp[s,4]/180*np.pi if grad_rotate_tmp else -1*mp[s,4].detach()/180*np.pi
        g = -1*mp[s,5]/180*np.pi if grad_rotate_tmp else -1*mp[s,5].detach()/180*np.pi

        transx = mp[s,0] if grad_translate_tmp else mp[s,0].detach()
        transy = mp[s,1] if grad_translate_tmp else mp[s,1].detach()
        transz = mp[s,2] if grad_translate_tmp else mp[s,2].detach()

        pshift = idx[:,None,None].repeat(1,len(idy),len(idz))*transx/x_dim 
        pshift += idy[None,:,None].repeat(len(idx),1,len(idz))*transy/y_dim
        pshift += idz[None,None:].repeat(len(idx),len(idy),1)*transz/z_dim
        pshift = torch.view_as_real(torch.exp(1j*2*np.pi*pshift)).cuda(args.gpu)
    
        trans = torch.zeros(3,3).cuda(args.gpu)
        trans[0,0] = torch.cos(a) * torch.cos(b)
        trans[0,1] = w1*(torch.cos(a) * torch.sin(b) * torch.sin(g) - torch.sin(a) * torch.cos(g))
        trans[0,2] = w2*( torch.cos(a) * torch.sin(b) * torch.cos(g) + torch.sin(a) * torch.sin(g))
        trans[1,0] = w3*(torch.sin(a) * torch.cos(b))
        trans[1,1] = torch.sin(a) * torch.sin(b) * torch.sin(g) + torch.cos(a) * torch.cos(g)
        trans[1,2] = w4*(torch.sin(a) * torch.sin(b) * torch.cos(g) - torch.cos(a) * torch.sin(g))
        trans[2,0] = -w5*(torch.sin(b))
        trans[2,1] = w6*(torch.cos(b) * torch.sin(g))
        trans[2,2] = torch.cos(b) * torch.cos(g)
        coord_rot = coord[:,idx_s,idy_s,:].reshape(3,-1).type(torch.FloatTensor)
        if s==0:
            rot_coord_sampled = trans@coord_rot.cuda(args.gpu)
            coord_idx = coord_rot.cuda(args.gpu)
            tran_vec = pshift[idx_s,idy_s,:,:].reshape(-1,2)
        else:
            rot_coord_sampled = torch.cat([rot_coord_sampled,trans@coord_rot.cuda(args.gpu)],dim=1)
            coord_idx = torch.cat([coord_idx,coord_rot.cuda(args.gpu)],dim=1)
            tran_vec = torch.cat([tran_vec,pshift[idx_s,idy_s,:,:].reshape(-1,2)],dim=0)
    
    rot_coord_sampled[0] = rot_coord_sampled[0]/x_dim*2*torch.pi
    rot_coord_sampled[1] = rot_coord_sampled[1]/y_dim*2*torch.pi
    rot_coord_sampled[2] = rot_coord_sampled[2]/z_dim*2*torch.pi
    # Using NUFFT to get the corrupted kspace
    nufftkb_forward.nufft.set_dims(len(rot_coord_sampled[0]), (kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]), 'cuda:'+str(args.gpu), Nb=3, norm=args.nufft_norm)
    nufftkb_forward.nufft.precompute(rot_coord_sampled.moveaxis(0,1))

    #nufftkb.nufft.set_dims(len(rot_coord_sampled[0]), (kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]), 'cuda:'+str(args.gpu), Nb=3, norm=None)
    #nufftkb.nufft.precompute(rot_coord_sampled.moveaxis(0,1))

    corrupted_kspace3D = torch.zeros_like(kspace3D).cuda(args.gpu)
    coord_idx[0] = torch.round(coord_idx[0]+x_dim//2)
    coord_idx[1] = torch.round(coord_idx[1]+y_dim//2)
    coord_idx[2] = torch.round(coord_idx[2]+z_dim//2)
    coord_idx = coord_idx.type(torch.long)
    #image3D_coil = ifft2c_ndim(kspace3D, 3)

    if max_coil_size is not None:
        coil_list = list(range(0,image3D_coil.shape[0]))
        coil_list_batches = [coil_list[i:i+max_coil_size] for i in range(0, len(coil_list), max_coil_size)]

        for jj,coil_batch in enumerate(coil_list_batches):
            if jj==0:
                ksp_corrupted_vec = nufftkb_forward.forward(rot_coord_sampled.moveaxis(0,1), image3D_coil[coil_batch])
            else:
                ksp_corrupted_vec = torch.cat([ksp_corrupted_vec, nufftkb_forward.forward(rot_coord_sampled.moveaxis(0,1), image3D_coil[coil_batch])],dim=0)
    else:
        ksp_corrupted_vec = nufftkb_forward.forward(rot_coord_sampled.moveaxis(0,1), image3D_coil)

    
    #if args.nufft_norm is None:
    eng_ratio = torch.sqrt(torch.sum(abs(ksp_corrupted_vec)**2)/torch.sum(abs(kspace3D[:,coord_idx[0],coord_idx[1],coord_idx[2],:])**2))
    if eng_ratio.requires_grad:
        eng_ratio = eng_ratio.detach()
    #else:
    #    eng_ratio = 1.0

    corrupted_kspace3D[:,coord_idx[0],coord_idx[1],coord_idx[2],:] = complex_mul(ksp_corrupted_vec/eng_ratio,tran_vec.unsqueeze(0))      

    ksp_corrupted_vec = None
    kspace3D = None
    image3D_coil = None
    rot_coord_sampled = None
    coord = None
    pshift = None
    grid_x = None
    grid_y = None
    grid_z = None

    return corrupted_kspace3D#, eng_ratio

