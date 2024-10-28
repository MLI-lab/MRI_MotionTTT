import torch
import torch.nn.functional as F
import numpy as np
import torchkbnufft as tkbn

from functions.utils.helpers.helpers_math import complex_mul

import functions.utils.motion_simulation.nufft.kbnufft as nufftkb_forward
import functions.utils.motion_simulation.nufft.kbnufft_2 as nufftkb_adjoint


def motion_correction_NUFFT(kspace3D, mp, traj, weight_rot, args, do_dcomp=True, num_iters_dcomp=3, grad_translate=True, grad_rotate=True, states_with_grad=None, shot_ind_maxksp=None, max_coil_size=None):
    '''
    Given a 3D k-space this function uses the adjoint NUFFT to compute the off-grid
    k-space values for the acquired lines specified in traj and with respect to the
    motion parameters mp. This k-space is then transformed back to image space.
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
        - grad_translate: Boolean, indicating whether to compute gradients for translations
        - grad_rotate: Boolean, indicating whether to compute gradients for rotations
        - states_with_grad: list of motion states for which gradients are computed
        - shot_ind_maxksp: a single index for which no gradients are computed
        - max_coil_size: maximum number of coils that are processed at once
    Output:
        - img3D: 3D tensor of shape (coils,x,y,z,2).    
    '''

    assert kspace3D.shape[-1] == 2, "Input k-space must be complex valued"
    if mp is None:
        mp = torch.zeros(len(traj[0]),6).cuda(args.gpu)
    else:
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

    if do_dcomp:
        dcomp = tkbn.calc_density_compensation_function(ktraj=rot_coord_sampled.detach(), 
                                                        im_size=(kspace3D.shape[1],kspace3D.shape[2],kspace3D.shape[3]),
                                                        num_iterations = num_iters_dcomp)
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
    
    # normalization based on energy ratio
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

def motion_corruption_NUFFT(kspace3D, image3D_coil, mp, traj, weight_rot, args, grad_translate=True, grad_rotate=True, states_with_grad=None, shot_ind_maxksp=None, max_coil_size=None):
    '''
    Given the 3D image volumen image3D_coil (potentially multiple coil imges) 
    this function uses the NUFFT to compute for each k-space line on the cartesian 
    grid defined in traj the off-grid k-space values from image3D_coil for the corresponding 
    motion state mp. The obtained values are placed on the coordinates specified by traj
    in the Cartesian corrupted k-space.
    Input:
        - kspace3D: 3D tensor of shape (coils,x,y,z,2) used for normalization
        - image3D_coil: 3D tensor of shape (coils,x,y,z,2) containing the image data
        input to the NUFFT
        - mp: motion parameters a tensor of shape (Ns, 6) with Ns the number of motion states
        and 6 the number of motion parameters (tx,ty,tz,alpha,beta,gamma). translations are in pixels
        and rotations in degrees.
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
        specifying which k-space lines were acquired under which motion state.
        - weight_rot: Boolean, if True, the rotation matrix is weighted to account
        for the aspect ratio of the image.
        - args: arguments of the experiment
        - grad_translate: Boolean, if True, gradients are computed for the translation parameters
        - grad_rotate: Boolean, if True, gradients are computed for the rotation parameters
        - states_with_grad: list of motion states for which gradients are computed
        - shot_ind_maxksp: a single index for which no gradients are computed
        - max_coil_size: maximum number of coils that are processed at once
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

    # normalization based on energy ratio
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


def rotate_translate_3D_complex_img(img3D, mp_align, weight_rot, gpu):
    x_dim, y_dim, z_dim = img3D.shape[1], img3D.shape[2], img3D.shape[3]
    w1 = y_dim/z_dim if weight_rot else 1
    w2 = x_dim/z_dim if weight_rot else 1
    w3 = z_dim/y_dim if weight_rot else 1
    w4 = x_dim/y_dim if weight_rot else 1
    w5 = z_dim/x_dim if weight_rot else 1
    w6 = y_dim/x_dim if weight_rot else 1

    a = mp_align[4]/180*np.pi
    b = mp_align[5]/180*np.pi
    g = mp_align[3]/180*np.pi

    # 
    trans = torch.zeros(3,4).cuda(gpu)
    trans[0,0] = torch.cos(a) * torch.cos(b)
    trans[0,1] = w1 * (torch.cos(a) * torch.sin(b) * torch.sin(g) - torch.sin(a) * torch.cos(g))
    trans[0,2] = w2 * ( torch.cos(a) * torch.sin(b) * torch.cos(g) + torch.sin(a) * torch.sin(g))
    trans[1,0] = w3 * (torch.sin(a) * torch.cos(b))
    trans[1,1] = torch.sin(a) * torch.sin(b) * torch.sin(g) + torch.cos(a) * torch.cos(g)
    trans[1,2] = w4 * (torch.sin(a) * torch.sin(b) * torch.cos(g) - torch.cos(a) * torch.sin(g))
    trans[2,0] = -w5 * (torch.sin(b))
    trans[2,1] = w6 * (torch.cos(b) * torch.sin(g))
    trans[2,2] = torch.cos(b) * torch.cos(g) 

    trans[0,3] = mp_align[2]/x_dim*2
    trans[1,3] = mp_align[0]/y_dim*2
    trans[2,3] = mp_align[1]/z_dim*2
    # Rotate/translate image
    if img3D.shape[-1]==2:
        img3D_real = img3D[...,0]
        img3D_imag = img3D[...,1]
        grid = F.affine_grid(trans[None, ...], img3D_real[None,...].shape, align_corners=False)
        img3D_real_rot = F.grid_sample(img3D_real[None,...], grid, align_corners=False, padding_mode='zeros', mode='bilinear')
        img3D_imag_rot = F.grid_sample(img3D_imag[None,...], grid, align_corners=False, padding_mode='zeros', mode='bilinear')
        img3D_rot = torch.stack([img3D_real_rot, img3D_imag_rot], dim=-1)[0,...]
        img3D_real = None
        img3D_imag = None
        img3D_real_rot = None
        img3D_imag_rot = None
    else:
        grid = F.affine_grid(trans[None, ...], img3D[None,...].shape, align_corners=False)
        img3D_rot = F.grid_sample(img3D[None,...], grid, align_corners=False, padding_mode='zeros', mode='bilinear')[0,...]
    
    grid = None

    return img3D_rot