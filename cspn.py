#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:37:41 2018

@author: Xinjing Cheng
@email : chengxinjing@baidu.com

"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

class Affinity_Propagate(nn.Module):
    
    def __init__(self, spn = False):
        super(Affinity_Propagate, self).__init__()
        self.spn = spn


    def forward(self, guidance, blur_depth, sparse_depth):
        
        # normalize features
        gate1_w1_cmb = torch.abs(guidance.narrow(1,0,1))
        gate2_w1_cmb = torch.abs(guidance.narrow(1,1,1))
        gate3_w1_cmb = torch.abs(guidance.narrow(1,2,1))
        gate4_w1_cmb = torch.abs(guidance.narrow(1,3,1))
        gate5_w1_cmb = torch.abs(guidance.narrow(1,4,1))
        gate6_w1_cmb = torch.abs(guidance.narrow(1,5,1))
        gate7_w1_cmb = torch.abs(guidance.narrow(1,6,1))
        gate8_w1_cmb = torch.abs(guidance.narrow(1,7,1))

        sparse_mask = sparse_depth.sign()
        
        result_depth = (1- sparse_mask)*blur_depth.clone()+sparse_mask*sparse_depth
        
        
        for i in range(16):
        # one propagation
            spn_kernel = 3 
            elewise_max_gate1 = self.eight_way_propagation(gate1_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate2 = self.eight_way_propagation(gate2_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate3 = self.eight_way_propagation(gate3_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate4 = self.eight_way_propagation(gate4_w1_cmb, result_depth, spn_kernel)  
            elewise_max_gate5 = self.eight_way_propagation(gate5_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate6 = self.eight_way_propagation(gate6_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate7 = self.eight_way_propagation(gate7_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate8 = self.eight_way_propagation(gate8_w1_cmb, result_depth, spn_kernel) 
            
            result_depth = self.max_of_8_tensor(elewise_max_gate1, elewise_max_gate2, elewise_max_gate3, elewise_max_gate4,\
                                                elewise_max_gate5, elewise_max_gate6, elewise_max_gate7, elewise_max_gate8)
            
            result_depth = (1- sparse_mask)*result_depth.clone()+sparse_mask*sparse_depth
    
        return result_depth
       
    
    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        [batch_size, channels, height, width] = weight_matrix.size()
        self.avg_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel, stride=1, padding=(kernel-1)/2, bias=False)
        weight = torch.ones(1, 1, kernel, kernel).cuda()
        weight[0,0,(kernel-1)/2,(kernel-1)/2]=0
        self.avg_conv.weight = nn.Parameter(weight)
        for param in self.avg_conv.parameters():
            param.requires_grad = False

        self.sum_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel, stride=1, padding=(kernel-1)/2, bias=False)
        sum_weight = torch.ones(1, 1, kernel, kernel).cuda()
        self.sum_conv.weight = nn.Parameter(sum_weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False
        weight_sum = self.sum_conv(weight_matrix)
        avg_sum = self.avg_conv((weight_matrix*blur_matrix))
        out = (torch.div(weight_matrix, weight_sum))*blur_matrix + torch.div(avg_sum, weight_sum)
        return out
        
    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1,0,1)
        gate1_x1_g2 = guidance.narrow(1,1,1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)  
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb
    
    
    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)    

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4) 
