
import logging
import torch
import os
from huggingface_hub import PyTorchModelHubMixin

from functions.utils.models.unet import Unet
from functions.utils.models.stackedUnet import StackedUnet
from functions.utils.helpers.helpers_init import init_optimization
from functions.utils.helpers.helpers_math import normalize_separate_over_ch


class MyModel(
    Unet,
    PyTorchModelHubMixin,
    pipeline_tag="image-to-image",
    license="mit",
):
    def __init__(self):
        super().__init__(in_chans = 2, out_chans=2, chans = 48, num_pool_layers = 4)

def get_model(args, verbose = True):
    
    if args.model == "unet":
        if verbose:
            logging.info(f"Using standard 2D Unet model.")
        model = Unet(in_chans=args.in_chans, 
                     out_chans=args.out_chans, 
                     chans=args.chans, 
                     num_pool_layers=args.pools)
    elif args.model == "stackedUnet":
        if verbose:
            logging.info(f"Using stacked Unet model with multiple input to single output.")
        model = StackedUnet(unet_num_ch_first_layer=args.chans, norm_type=args.norm_type_unet)

    if args.train:
        optimizer, scheduler =  init_optimization(args, model, "train")
    else:
        optimizer, scheduler = None, None

    if args.load_model_from_huggingface != None:
        model = MyModel.from_pretrained(args.load_model_from_huggingface)
        if verbose:
            logging.info(f"Loaded model from huggingface {args.load_model_from_huggingface}")
        current_epoch = 0

    elif args.load_model_path != None:
        checkpoint_loaded = torch.load(args.load_model_path, map_location=torch.device(f"cuda:{args.gpu}"))
        model = model.cuda(args.gpu)
        model.load_state_dict(checkpoint_loaded['model'])
        if args.train_load_optimizer_scheduler:
            optimizer.load_state_dict(checkpoint_loaded["optimizer"])
            scheduler.load_state_dict(checkpoint_loaded["scheduler"])
            current_epoch = checkpoint_loaded["epoch"]
            if verbose:
                logging.info(f"Loaded model, optimizer, scheduler from {args.load_model_path}")
        else:
            if verbose:
                logging.info(f"Loaded model from {args.load_model_path}")
            current_epoch = 0
        del checkpoint_loaded
    else:
        current_epoch = 0

    if verbose:
        logging.info(f"Built a {args.model} consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    model = model.cuda(args.gpu)

    return model, optimizer, scheduler, current_epoch


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