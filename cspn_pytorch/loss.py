# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:06:56 2018

@author: norbot
"""

import torch
import torch.nn as nn


class Wighted_L1_Loss(torch.nn.Module):
    def __init__(self):
        super(Wighted_L1_Loss, self).__init__()

    def forward(self, pred, label):
        label_mask = label > 0.0001
        _pred = pred[label_mask]
        _label = label[label_mask]
        n_valid_element = _label.size(0)
        diff_mat = torch.abs(_pred-_label)
        loss = torch.sum(diff_mat)/n_valid_element
        return loss
