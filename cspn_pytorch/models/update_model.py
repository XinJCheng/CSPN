"""
Created on Mon Feb  5 16:19:25 2018

@author: norbot
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import string

# update pretrained model params according to my model params
def update_model(my_model, pretrained_dict):
    my_model_dict = my_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict}
    # 2. overwrite entries in the existing state dict
    my_model_dict.update(pretrained_dict)

    return my_model_dict

# dont know why my offline saved model has 'module.' in front of all key name
def remove_moudle(remove_dict):
    for k, v in remove_dict.items():
        if 'module' in k :
            print("==> model dict with addtional module, remove it...")
            removed_dict = { k[7:]: v for k, v in remove_dict.items()}
        else:
            removed_dict = remove_dict
        break
    return removed_dict

def update_conv_spn_model(out_dict, in_dict):
    in_dict = {k: v for k, v in in_dict.items() if k in out_dict}
    return in_dict

