"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


from functions.helpers.helpers_math import complex_abs, center_crop

l1_loss_sum = torch.nn.L1Loss(reduction='sum')
l2_loss_mean = torch.nn.MSELoss(reduction='mean')


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, gpu: int = 0):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.win_size = win_size
        self.k1, self.k2 = torch.tensor(k1).cuda(gpu), torch.tensor(k2).cuda(gpu)
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size).cuda(gpu) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = torch.tensor(NP / (NP - 1)).cuda(gpu)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()



def get_train_loss_function(args):
    
        if args.train_loss == "joint":
            train_loss_function = joint_train_loss
            logging.info(f"Using joint_train_loss without masking the model output function")

        elif args.train_loss == "sup_mag":
            train_loss_function = sup_mag_train_loss
            logging.info(f"Using sup_mag_train_loss without masking the model output function")

        elif args.train_loss == "sup_ksp":
            train_loss_function = sup_ksp_train_loss
            logging.info(f"Using sup_ksp_train_loss function")
            
        elif args.train_loss == "sup_compimg":
                train_loss_function = sup_compimg_train_loss
                logging.info(f"Using sup_compimg_train_loss without masking the model output function")
        else:
            raise ValueError(f"Unknown train loss function: {args.train_loss}")
    
        return train_loss_function

#####################################################
# Define all possible training loss functions

def sup_ksp_train_loss(target_kspace, recon_kspace_full, target_image_full_2c, recon_image_full_2c):

    loss_ksp = l1_loss_sum(recon_kspace_full, target_kspace) / torch.sum(torch.abs(target_kspace))

    loss_img = torch.tensor(0.0)

    loss = loss_img + loss_ksp
    return loss, loss_img, loss_ksp

def sup_compimg_train_loss(target_kspace, recon_kspace_full, target_image_full_2c, recon_image_full_2c):

    loss_img = l1_loss_sum(recon_image_full_2c, target_image_full_2c) / torch.sum(torch.abs(target_image_full_2c))

    loss_ksp = torch.tensor(0.0)

    loss = loss_img + loss_ksp
    return loss, loss_img, loss_ksp   


def joint_train_loss(target_kspace, recon_kspace_full, target_image_full_2c, recon_image_full_2c):

    loss_ksp = l1_loss_sum(recon_kspace_full, target_kspace) / torch.sum(torch.abs(target_kspace))

    recon_image_full_1c = complex_abs(recon_image_full_2c+1e-9)
    target_image_full_1c = complex_abs(target_image_full_2c+1e-9)
    loss_img = l1_loss_sum(recon_image_full_1c, target_image_full_1c) / torch.sum(torch.abs(target_image_full_1c))

    loss = loss_img + loss_ksp
    return loss, loss_img, loss_ksp


def sup_mag_train_loss(target_kspace, recon_kspace_full, target_image_full_2c, recon_image_full_2c):

    recon_image_full_1c = complex_abs(recon_image_full_2c+1e-9)
    target_image_full_1c = complex_abs(target_image_full_2c+1e-9)
    loss_img = l1_loss_sum(recon_image_full_1c, target_image_full_1c) / torch.sum(torch.abs(target_image_full_1c))

    loss_ksp = torch.tensor(0.0)

    loss = loss_img + loss_ksp
    return loss, loss_img, loss_ksp



