
import torch
from typing import Optional, Sequence, Tuple, Union

from functions.helpers.helpers_math import complex_abs

def PSNR_torch(recon_img: torch.Tensor, ref_img: torch.Tensor):
    
    mse = torch.mean((recon_img - ref_img)**2)
    psnr = 20 * torch.log10(torch.tensor(ref_img.max().item()))- 10 * torch.log10(mse)

    return psnr.item()