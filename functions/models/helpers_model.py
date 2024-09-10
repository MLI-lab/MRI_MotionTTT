
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
        model.load_state_dict(torch.load(checkpoint_path_for_loading, map_location=torch.device('cpu')))
        logging.info(f"Loaded model from {checkpoint_path_for_loading}")


    elif args.load_external_model_path != "None" and args.load_1ch_model:

        logging.info(f"Load pretrained model from {args.load_external_model_path}")

        # for param in model.parameters():
        #     print(param.shape, param.requires_grad)
        #     if len(param.shape) == 4:
        #         print(param[1,1,0,:])
        
        cp = torch.load(args.load_external_model_path, map_location='cpu')
        for key in cp['model_state_dict'].keys():
            #print(key, cp['model_state_dict'][key].shape)
            
            if model.state_dict()[key[5:]].shape == cp['model_state_dict'][key].shape:
                # replace the state_dict of the 2ch model with the 1ch model
                model.state_dict()[key[5:]].copy_(cp['model_state_dict'][key])
                
                #model.state_dict()[key[5:]] = cp['model_state_dict'][key] # kangs unets have a prefix unet. in the state_dict keys
                
            else:
                if key[5:] == 'down_sample_layers.0.layers.0.weight':
                    model.state_dict()[key[5:]][:,0:1,:,:].copy_(cp['model_state_dict'][key])
                    model.state_dict()[key[5:]][:,1:2,:,:].copy_(cp['model_state_dict'][key])
                    logging.info(f"Copying first channel of {key} from 1ch model to both channels of 2ch model")
                elif key[5:] == 'up_conv.3.1.weight':
                    model.state_dict()[key[5:]][0:1,:,:,:].copy_(cp['model_state_dict'][key])
                    model.state_dict()[key[5:]][1:2,:,:,:].copy_(cp['model_state_dict'][key])
                    logging.info(f"Copying first channel of {key} from 1ch model to both channels of 2ch model")
                elif key[5:] == 'up_conv.3.1.bias':
                    model.state_dict()[key[5:]][0:1].copy_(cp['model_state_dict'][key])
                    model.state_dict()[key[5:]][1:2].copy_(cp['model_state_dict'][key])
                    logging.info(f"Copying first channel of {key} from 1ch model to both channels of 2ch model")
                else:
                    raise Exception(f"Shape mismatch 2ch model has {model.state_dict()[key[5:]].shape} vs 1ch model has {cp['model_state_dict'][key].shape}")
                

                #logging.info(f"Skipping key {key} due to shape mismatch 2ch model has {model.state_dict()[key[5:]].shape} vs 1ch model has {cp['model_state_dict'][key].shape}")

        # Set all gradients to zero except first and last layer
        if args.train_only_first_last_layers:
            for name, param in model.named_parameters():
                if name in ['down_sample_layers.0.layers.0.weight', 'up_conv.3.1.weight', 'up_conv.3.1.bias', 
                            'down_sample_layers.0.layers.4.weight', 'up_conv.3.0.layers.4.weight']:
                    param.requires_grad = True
                    logging.info(f"Set grad of {name} with shape {param.shape} to True")
                else:
                    param.requires_grad = False


        # for param in model.parameters():
        #     print(param.shape, param.requires_grad)
        #     if len(param.shape) == 4:
        #         print(param[1,1,0,:])
                
        

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