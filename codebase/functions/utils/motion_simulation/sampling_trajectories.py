import torch
import numpy as np
import logging
import os
import pickle

from functions.utils.helpers.helpers_math import  chunks

from functions.utils.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories


def sim_motion_get_traj(args, mask3D, verbose=True):

    if args.sampTraj_simMot == "interleaved_cartesian":
        logging.info(f"Generate interleaved cartesian sampling trajectory with center_in_first_state {args.center_in_first_state}") if verbose else None
        traj, _ = generate_interleaved_cartesian_trajectory(args.Ns, mask3D, args, center_in_first_state=args.center_in_first_state)

    elif args.sampTraj_simMot == "random_cartesian":
        logging.info(f"Generate random cartesian sampling trajectory with center_in_first_state {args.center_in_first_state}") if verbose else None
        traj, _ = generate_random_cartesian_trajectory(args.Ns, mask3D, args, center_in_first_state=args.center_in_first_state, seed=args.random_sampTraj_seed)

    elif args.sampTraj_simMot == "deterministic_cartesian":
        logging.info(f"Load deterministic cartesian sampling trajectory from {args.sampling_order_path}") if verbose else None
        traj, _ = generate_deterministic_cartesian_trajectory(args.Ns, mask3D, args, path_to_traj=args.sampling_order_path)

    elif args.sampTraj_simMot == "linear_cartesian":
        logging.info(f"Generate linear cartesian sampling trajectory.") if verbose else None
        traj, _ = generate_linear_cartesian_trajectory(args.Ns, mask3D, args)
    else:
        raise ValueError(f"sampTraj_simMot {args.sampTraj_simMot} not implemented.")

    return traj


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

    with open(path_to_traj,'rb') as fn:
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

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states

def generate_linear_cartesian_trajectory(Ns, mask3D, args=None, save_path=None, dir='y'):
    '''
    Given a 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1) the 
    acquired k-space lines are specified by the phase_enc1 and phase_enc2 plane.
    The acquired lines are assgined to the motion states in a linear fashion.
    Hence, if Ns=10 the first 10th fraction of lines ordered along the axis 
    specified by dir are assigned to the first motion state and so on.
    Input:
        - Ns: number of motion states
        - mask3D: 3D mask of shape (coils,phase_enc1,phase_enc2,freq_enc,1)
        - args: arguments of the experiment
        - save_path: path to save the masks
        - dir: direction of the sampling trajectory (either 'x' or 'y' 
        corresponding to phase_enc1 or phase_enc2)      
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

    if dir=='x':
        traj = (list(chunks(y_coord,Ns)), list(chunks(x_coord,Ns)))
    else:
        traj = (list(chunks(x_coord,Ns)), list(chunks(y_coord,Ns)))

    # For each pair of x/y-coordinates in traj[0][i]/traj[1][i] sort the x/y-coordinates according to x coordinates in traj[0][i] in ascending order
    for i in range(Ns):
        traj[0][i], traj[1][i] = zip(*sorted(zip(traj[0][i],traj[1][i])))

        

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states


