import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def continue_training(checkpoint_path, generator: DDP, mpd: DDP, mrd: DDP, optimizer_d: optim.Optimizer, optimizer_g: optim.Optimizer) -> int:
    """load the latest checkpoints and optimizers"""
    generator_dict = {}
    mpd_dict = {}
    mrd_dict = {}
    optimizer_d_dict = {}
    optimizer_g_dict = {}
    
    # globt all the checkpoints in the directory
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pt"):
            name, epoch_str = file.rsplit('_', 1)
            epoch = int(epoch_str.split('.')[0])
            
            if name.startswith("generator"):
                generator_dict[epoch] = file
            elif name.startswith("mpd"):
                mpd_dict[epoch] = file
            elif name.startswith("mrd"):
                mrd_dict[epoch] = file
            elif name.startswith("optimizerd"):
                optimizer_d_dict[epoch] = file
            elif name.startswith("optimizerg"):
                optimizer_g_dict[epoch] = file
    
    # get the largest epoch
    common_epochs = set(generator_dict.keys()) & set(mpd_dict.keys()) & set(mrd_dict.keys()) & set(optimizer_d_dict.keys()) & set(optimizer_g_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        generator_path = os.path.join(checkpoint_path, generator_dict[max_epoch])
        mpd_path = os.path.join(checkpoint_path, mpd_dict[max_epoch])
        mrd_path = os.path.join(checkpoint_path, mrd_dict[max_epoch])
        optimizer_d_path = os.path.join(checkpoint_path, optimizer_d_dict[max_epoch])
        optimizer_g_path = os.path.join(checkpoint_path, optimizer_g_dict[max_epoch])
        
        # load model and optimizer
        generator.module.load_state_dict(torch.load(generator_path, map_location='cpu'))
        mpd.module.load_state_dict(torch.load(mpd_path, map_location='cpu'))
        mrd.module.load_state_dict(torch.load(mrd_path, map_location='cpu'))
        optimizer_d.load_state_dict(torch.load(optimizer_d_path, map_location='cpu'))
        optimizer_g.load_state_dict(torch.load(optimizer_g_path, map_location='cpu'))
        
        print(f'resume model and optimizer from {max_epoch} epoch')
        return max_epoch + 1
    
    else:
        return 0