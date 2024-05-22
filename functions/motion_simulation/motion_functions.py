import torch
import torch.nn.functional as F
import numpy as np
import torchkbnufft as tkbn
import time

from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, normalize_separate_over_ch
from functions.helpers.helpers_math import complex_abs, complex_mul, norm_to_gt, ifft2c_ndim, fft2c_ndim, complex_conj

#from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex

from functions.helpers.helpers_log_save_image_utils import save_slice_images_from_volume, volume_to_k3d_html
from functions.helpers.helpers_log_save_image_utils import save_masks_from_motion_sampling_trajectories

#import functions.motion_simulation.kbnufft as nufftkb
import functions.motion_simulation.kbnufft as nufftkb_forward
import functions.motion_simulation.kbnufft_2 as nufftkb_adjoint

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
    align_final = torch.zeros(6).cuda(gpu)
    for i in range(6):
        align_set = np.linspace(base_align[i]-r,base_align[i]+r,num_points)
        motion_mae_total = []
        for align in align_set:
            mp_est_align=mp_pred[:,i]-align
            motion_mae_total.append(abs(mp_est_align-mp_gt[:,i]).mean().item())
        align_final[i] = align_set[np.argmin(np.array(motion_mae_total))]
        # print(f'{i+1}/{6} Finished')
    return mp_pred - align_final

def gen_rand_mot_params_eventModel(Ns, max_trans, max_rot, seed, num_events):
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
    Output:
        - motion_params: tensor of shape (Ns, 6) with the motion parameters    
    '''
    motion_params = torch.zeros(Ns, 6)
    motion_params_events = torch.zeros(num_events, 6)
    torch.manual_seed(seed)
    motion_params_events[:,0:3] = torch.rand([num_events,3]) * 2 * max_trans - max_trans
    motion_params_events[:,3:6] = torch.rand([num_events,3]) * 2 * max_rot - max_rot

    # pick random motion states for the events and sort them
    event_states = torch.randperm(Ns)[:num_events]
    event_states = torch.sort(event_states)[0]
    for i in range(len(event_states)):
        if i == len(event_states)-1:
            motion_params[event_states[i]:,:] = motion_params_events[i:i+1,:]
        else:
            motion_params[event_states[i]:event_states[i+1],:] = motion_params_events[i:i+1,:]


    return motion_params



def generate_random_motion_params(Ns, max_trans, max_rot, seed):
    '''
    Genereate Ns-many tuples of length 6 containing 3 translations uniform in 
    [-max_transl, max_transl] and 3 rotations uniform in [-max_rot, max_rot].
    Input:
        - Ns: number of motion states
        - max_trans: maximum translation in pixels
        - max_rot: maximum rotation in degrees
        - seed: random seed
    Output:
        - motion_params: tensor of shape (Ns, 6) with the motion parameters    
    '''
    motion_params = torch.zeros(Ns, 6)
    torch.manual_seed(seed)
    motion_params[:,0:3] = torch.rand([Ns,3]) * 2 * max_trans - max_trans
    motion_params[:,3:6] = torch.rand([Ns,3]) * 2 * max_rot - max_rot

    return motion_params

def generate_interleaved_cartesian_trajectory(Ns, mask3D, args=None, save_path=None):
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
    Output:
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    '''
    assert len(mask3D.shape) == 5, "Mask must have shape (coils,phase_enc1,phase_enc2,freq_enc,1)"

    mask2D = mask3D[0,:,:,0,0].cpu().numpy() 
    mask2D_center = np.zeros_like(mask2D)
    mask2D_center[mask2D.shape[0]//2-1:mask2D.shape[0]//2+2,mask2D.shape[1]//2-1:mask2D.shape[1]//2+2] = 1
    mask2D_no_center = mask2D - mask2D_center
    
    # assign 3x3 center lines to the same motion state
    recordedx_center, recordedy_center = np.where(mask2D_center==1)
    recordedx, recordedy = np.where(mask2D_no_center==1)

    traj = ([recordedx[i:len(recordedx):Ns] for i in range(Ns)], [recordedy[i:len(recordedy):Ns] for i in range(Ns)])
    
    # attach the center lines to the trajectory of the first motion state
    traj[0][0] = np.concatenate((traj[0][0], recordedx_center))
    traj[1][0] = np.concatenate((traj[1][0], recordedy_center))

    if save_path:
        masks2D_all_states = torch.from_numpy(save_masks_from_motion_sampling_trajectories(traj, mask2D, save_path=save_path, save_figures=False)).cuda(args.gpu)
    else:
        masks2D_all_states = None

    return traj, masks2D_all_states



def unet_forward(net, masked_corrected_img3D, rec_id):
    '''
    Forward map of the UNet
    Input:
    * net: the trained net for motion-free data reconstruction
    * kspace3d: 3d kspace
    * sens_maps: 3d sensitivity maps
    * rec_id: the slices that we need to reconsturct
    '''
    # Move axial dim to the first dim (batch dim)
    masked_corrected_img3D = torch.moveaxis(masked_corrected_img3D,2,0)    
    inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(masked_corrected_img3D,-1,1), eps=1e-11)
    
    # Reconstruct the NN with no grad:
    with torch.no_grad():
        image_recon = net(inputs_img_full_2c_norm)
    # Reconstruct image using grad for defined slices
    image_recon[rec_id] = net(inputs_img_full_2c_norm[rec_id])
    
    image_recon = image_recon * std + mean   
    image_recon = torch.moveaxis(image_recon,1,-1)    # (batch, x, y, ch)
    image_recon = torch.moveaxis(image_recon,0,2)    
    #torch.cuda.empty_cache()
    return image_recon

def unet_forward_all_axes(net, masked_corrected_img3D, rec_id, ax_id):
    '''
    Forward map of the UNet
    Input:
    * net: the trained net for motion-free data reconstruction
    * kspace3d: 3d kspace
    * sens_maps: 3d sensitivity maps
    * rec_id: the slices that we need to reconsturct
    '''
    # Move axial dim to the first dim (batch dim)
    masked_corrected_img3D = torch.moveaxis(masked_corrected_img3D,ax_id,0)    
    inputs_img_full_2c_norm, mean, std = normalize_separate_over_ch(torch.moveaxis(masked_corrected_img3D,-1,1), eps=1e-11)
    
    # Reconstruct the NN with no grad:
    with torch.no_grad():
        image_recon = net(inputs_img_full_2c_norm)
    # Reconstruct image using grad for defined slices
    image_recon[rec_id] = net(inputs_img_full_2c_norm[rec_id])
    
    image_recon = image_recon * std + mean   
    image_recon = torch.moveaxis(image_recon,1,-1)    # (batch, x, y, ch)
    image_recon = torch.moveaxis(image_recon,0,ax_id)    
    #torch.cuda.empty_cache()
    return image_recon


def motion_correction_NUFFT(kspace3D, mp, traj, weight_rot, args, do_dcomp=True, num_iters_dcomp=3, grad_translate=True, grad_rotate=True, states_with_grad=None, max_coil_size=None):
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
    Output:
        - img3D: 3D tensor of shape (coils,x,y,z,2).    
    '''

    assert kspace3D.shape[-1] == 2, "Input k-space must be complex valued"
    if mp is not None:
        assert mp.shape[0]+1 == len(traj[0]), "Number of motion states minus one must match number of trajectory states"
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

    # copy k-space lines from first motion state to the sampled and translated k-space, which
    # is the input to the nufft
    ksp_sampled = kspace3D[:,IDx[0],IDy[0],:,:]
    ksp_sampled = torch.view_as_complex(ksp_sampled).reshape(kspace3D.shape[0],-1)
    rot_coord_sampled = coord[:,IDx[0],IDy[0],:].reshape(3,-1).type(torch.FloatTensor).cuda(args.gpu)

    for s in range(Ns-1):

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

        idx_s = IDx[s+1]
        idy_s = IDy[s+1]
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

        coord_rot = coord[:,idx_s,idy_s,:].reshape(3,-1).type(torch.FloatTensor)
        rot_coord_sampled = torch.cat([rot_coord_sampled,trans@coord_rot.cuda(args.gpu)],dim=1)

    rot_coord_sampled[0] = rot_coord_sampled[0]/x_dim*2*torch.pi
    rot_coord_sampled[1] = rot_coord_sampled[1]/y_dim*2*torch.pi
    rot_coord_sampled[2] = rot_coord_sampled[2]/z_dim*2*torch.pi


    nufftkb_adjoint.nufft.set_dims(len(rot_coord_sampled[0]), (kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]), 'cuda:'+str(args.gpu), Nb=3, norm=args.nufft_norm)
    nufftkb_adjoint.nufft.precompute(rot_coord_sampled.moveaxis(0,1))

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
        
        
    eng_ratio = torch.sqrt(torch.sum(abs(img3D)**2)/torch.sum(abs(ksp_sampled)**2))
    if eng_ratio.requires_grad:
        eng_ratio = eng_ratio.detach()


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

    return img3D

def motion_corruption_NUFFT(kspace3D, image3D_coil, mp, traj, weight_rot, args, grad_translate=True, grad_rotate=True, states_with_grad=None, max_coil_size=None):
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
    #assert mp.shape[0]+1 == len(traj[0]), "Number of motion states minus one must match number of trajectory states"
    assert len(kspace3D.shape) == 5, "Input k-space must have shape (coils,x,y,z,2)"

    if mp.shape[0]+1 == len(traj[0]):
        zero_state_is_present = True
    elif mp.shape[0] == len(traj[0]):
        zero_state_is_present = False
    else:
        raise ValueError("Number of motion states (optionally minus one) must match number of trajectory states")


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
            if s+1 in states_with_grad:
                grad_translate_tmp = grad_translate
                grad_rotate_tmp = grad_rotate
            else:
                grad_translate_tmp = False
                grad_rotate_tmp = False
        else:
            grad_translate_tmp = grad_translate
            grad_rotate_tmp = grad_rotate

        if s==0 and zero_state_is_present:
            a=torch.tensor(0)
            b=torch.tensor(0)
            g=torch.tensor(0)
        else:
            a = -1*mp[s-1,3]/180*np.pi if grad_rotate_tmp else -1*mp[s-1,3].detach()/180*np.pi
            b = -1*mp[s-1,4]/180*np.pi if grad_rotate_tmp else -1*mp[s-1,4].detach()/180*np.pi
            g = -1*mp[s-1,5]/180*np.pi if grad_rotate_tmp else -1*mp[s-1,5].detach()/180*np.pi

        if s==0 and zero_state_is_present:
            transx = 0
            transy = 0
            transz = 0
        else:
            transx = mp[s-1,0] if grad_translate_tmp else mp[s-1,0].detach()
            transy = mp[s-1,1] if grad_translate_tmp else mp[s-1,1].detach()
            transz = mp[s-1,2] if grad_translate_tmp else mp[s-1,2].detach()

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


    corrupted_kspace3D = torch.zeros_like(kspace3D).cuda(args.gpu)
    coord_idx[0] = torch.round(coord_idx[0]+x_dim//2)
    coord_idx[1] = torch.round(coord_idx[1]+y_dim//2)
    coord_idx[2] = torch.round(coord_idx[2]+z_dim//2)
    coord_idx = coord_idx.type(torch.long)

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

    
    eng_ratio = torch.sqrt(torch.sum(abs(ksp_corrupted_vec)**2)/torch.sum(abs(kspace3D[:,coord_idx[0],coord_idx[1],coord_idx[2],:])**2))
    if eng_ratio.requires_grad:
        eng_ratio = eng_ratio.detach()


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

    return corrupted_kspace3D

