#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:32:49 2018

@author: Xinjing Cheng
@email : chengxinjing@baidu.com

"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
import time
import torch.utils.model_zoo as model_zoo
import post_process as post_process
from torch.autograd import Variable
import update_model

# memory analyze
import gc

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_path ={
    'resnet18': 'pretrained/resnet18.pth',
    'resnet50': 'pretrained/resnet50.pth'
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale):
        oheight = 0
        owidth = 0
        if self.oheight == 0 and self.owidth == 0:
            oheight = scale * x.size(2)
            owidth = scale * x.size(3)
            x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        else:
            oheight = self.oheight
            owidth = self.owidth
            x = nn.Upsample(size=(oheight, owidth), mode='nearest')(x)
        mask = torch.zeros_like(x)
        for h in range(0, oheight, 2):
            for w in range(0, owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Simple_Gudi_UpConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale):
        
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x.narrow(2,0,self.oheight)
            x = x.narrow(3,0,self.owidth)
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        return out

class Simple_Gudi_UpConv_Block_Last_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.oheight = oheight
        self.owidth = owidth

    def _up_pooling(self, x, scale):

        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:, :, h, w] = 1
        x = torch.mul(mask, x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out

class Gudi_UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale):
        
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x[:,:,0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale):
        
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x[:,:,0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x, side_input):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, up_proj_block):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.mid_channel = 256*block.expansion
        self.conv2 = nn.Conv2d(512*block.expansion, 512*block.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512*block.expansion)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.post_process_layer = self._make_post_process_layer()
        
        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 15, 19)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 29, 38)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 57, 76)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 114, 152)
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 228, 304)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 8, 228, 304)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)
    
    def _make_post_process_layer(self):
        return post_process.Affinity_Propagate(spn=True)
    
    def forward(self, x):
        [batch_size, channel, height, width] = x.size()
        sparse_depth = x.narrow(1,3,1).clone()
        
        x = self.conv1_1(x)
        skip4 = x
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x
        
        x = self.layer2(x)
        skip2 = x
        
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(self.conv2(x))
        x = self.gud_up_proj_layer1(x)
        
        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)
        x= self.gud_up_proj_layer5(x)
        
        guidance = self.gud_up_proj_layer6(x)
        x = self.post_process_layer(guidance, x, sparse_depth)
        
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model..')
        pretrained_dict = torch.load(model_path['resnet18'])
        model.load_state_dict(update_model.update_model(model, pretrained_dict))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], UpProj_Block, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model..')
        pretrained_dict = torch.load(model_path['resnet50'])
        model.load_state_dict(update_model.update_model(model, pretrained_dict))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], UpProj_Block, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], UpProj_Block, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

