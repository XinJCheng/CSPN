"""
@author: Xinjing Cheng & Peng Wang

"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Affinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time,
                 prop_kernel,
                 norm_type='8sum'):
        """

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1


    def forward(self, guidance, blur_depth, sparse_depth=None):
        '''
        guidance is the 8xhxw feature map representing 8 directional propagation weight
        blur_depth is the output of deconv as the final depth
        sparse_depth is the original sparse depth map
        '''
        self.sum_conv = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(1, 1, 1),
                                  stride=1,
                                  padding=0,
                                  bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).cuda()
        self.sum_conv.weight = nn.Parameter(weight)
        # no need to update the weight here, since we update the weights equally to all directions
        for param in self.sum_conv.parameters():
            param.requires_grad = False

        gate_wb, gate_sum = self.affinity_normalization(guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth

        #blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()

        for i in range(self.prop_time):
            # one propagation
            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)
        # guidence here is batch_size x 8 x h x w
        # gate1_wb_cmb here is batch_size x 1 x h x w
        # its taking 8 feature maps as the 8 directional propagation guidance map
        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # gate1_wb_cmb before zero pad 2d is: torch.Size([1, 1, 228, 912])
        # gate1_wb_cmb after zero pad 2d is:
        # Out[17]: torch.Size([1, 1, 1, 230, 914])
        # top pad
        # ZeroPad2d((left, right, top, bot))
        '''
        this will create a padding on the bottom right:
        a = torch.rand(3,3)
        left_top_pad(a).unsqueeze(1)
        Out[10]: 
        tensor([[[0.5680, 0.8088, 0.0461, 0.0000, 0.0000]],
                [[0.5188, 0.9142, 0.7045, 0.0000, 0.0000]],
                [[0.3563, 0.7770, 0.1678, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
        '''
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)
        # same here:
        '''
        center_top_pad(a).unsqueeze(1)
        Out[11]: 
        tensor([[[0.0000, 0.5680, 0.8088, 0.0461, 0.0000]],
            [[0.0000, 0.5188, 0.9142, 0.7045, 0.0000]],
            [[0.0000, 0.3563, 0.7770, 0.1678, 0.0000]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
        '''
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)
        '''
        right_top_pad(gate3_wb_cmb).unsqueeze(1)
        Out[13]: 
        tensor([[[[[[ 0.0000,  0.0000,  0.0000,  ...,  2.1156,  0.0158,  0.8841],
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0829,  3.4573, -0.1354],
                    [ 0.0000,  0.0000,  0.0000,  ...,  2.7383,  0.0282,  1.0868],
                    ...,
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]]]],
               device='cuda:0', grad_fn=<UnsqueezeBackward0>)
        '''
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad

        '''
        left_center_pad(a).unsqueeze(1)
        Out[14]: 
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.5680, 0.8088, 0.0461, 0.0000, 0.0000]],
                [[0.5188, 0.9142, 0.7045, 0.0000, 0.0000]],
                [[0.3563, 0.7770, 0.1678, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
        '''

        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)
        '''
        right_center_pad(a).unsqueeze(1)
        Out[15]: 
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.5680, 0.8088, 0.0461]],
                [[0.0000, 0.0000, 0.5188, 0.9142, 0.7045]],
                [[0.0000, 0.0000, 0.3563, 0.7770, 0.1678]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
        '''
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        '''
        left_bottom_pad(a).unsqueeze(1)
        Out[16]: 
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.5680, 0.8088, 0.0461, 0.0000, 0.0000]],
                [[0.5188, 0.9142, 0.7045, 0.0000, 0.0000]],
                [[0.3563, 0.7770, 0.1678, 0.0000, 0.0000]]])
        '''
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)
        '''
        center_bottom_pad(a).unsqueeze(1)
        Out[19]: 
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.5680, 0.8088, 0.0461, 0.0000]],
                [[0.0000, 0.5188, 0.9142, 0.7045, 0.0000]],
                [[0.0000, 0.3563, 0.7770, 0.1678, 0.0000]]])
        '''
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)
        '''
        right_bottm_pad(a).unsqueeze(1)
        Out[20]: 
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.5680, 0.8088, 0.0461]],
                [[0.0000, 0.0000, 0.5188, 0.9142, 0.7045]],
                [[0.0000, 0.0000, 0.3563, 0.7770, 0.1678]]])
        '''
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)
        '''
        gate_wb.shape
        Out[21]: torch.Size([1, 8, 1, 230, 914])
        '''
        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        '''
        sum will get all 8 direction sum of abs_weight:
        abs_weight.shape
        Out[25]: torch.Size([1, 1, 1, 230, 914])
        '''
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        # normalize over 8 directions
        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        # so this basically move the image from center to 8 directions for 2 cells and add them up
        # then divide, now it cut into the original image shape.
        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum


    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth


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


