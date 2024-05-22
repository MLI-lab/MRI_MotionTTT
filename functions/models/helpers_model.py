
import logging
import torch
import os

from functions.models.unet import Unet
from functions.helpers.helpers_init import init_optimization


def get_model(args):


    if args.model == "unet":
        model = Unet(in_chans=args.in_chans, 
                     out_chans=args.out_chans, 
                     chans=args.chans, 
                     num_pool_layers=args.pools, 
                     drop_prob=0.0)
        
    optimizer, scheduler =  init_optimization(args, model, "train")

    if args.load_model_from_checkpoint != "None" and args.load_external_model_path == "None":

        checkpoint_path_for_loading = os.path.join(args.checkpoint_path, args.load_model_from_checkpoint)

        checkpoint_loaded = torch.load(checkpoint_path_for_loading, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint_loaded['model'])
        if args.train_load_optimizer_scheduler:
            optimizer.load_state_dict(checkpoint_loaded["optimizer"])
            scheduler.load_state_dict(checkpoint_loaded["scheduler"])
            logging.info(f"Loaded model, optimizer, scheduler from {args.load_external_model_path}")
        else:
            logging.info(f"Loaded model from {args.load_external_model_path}")


    elif args.load_external_model_path != "None":
        checkpoint_loaded = torch.load(args.load_external_model_path, map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint_loaded["model"][0])
        model.load_state_dict(checkpoint_loaded['model'])
        if args.train_load_optimizer_scheduler:
            optimizer.load_state_dict(checkpoint_loaded["optimizer"])
            scheduler.load_state_dict(checkpoint_loaded["scheduler"])
            logging.info(f"Loaded model, optimizer, scheduler from {args.load_external_model_path}")
        else:
            logging.info(f"Loaded model from {args.load_external_model_path}")

    logging.info(f"Built a {args.model} consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    model = model.cuda(args.gpu)

    return model, optimizer, scheduler