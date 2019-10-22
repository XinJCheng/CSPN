#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:07:52 2018

@author: norbot
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import data_transform
from PIL import Image, ImageOps
import h5py

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
imagenet_eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

class KittiDataset(Dataset):
    # nyu depth dataset 
    def __init__(self, csv_file, root_dir, split, n_sample=500, input_format = 'img'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rgbd_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.input_format = input_format
        self.n_sample = n_sample
    
    def __len__(self):
        return len(self.rgbd_frame)

    def __getitem__(self, idx):
        # read input image
        if self.input_format == 'img':
#            print('==> Input Format is image')
            rgb_name = os.path.join(self.root_dir,
                                    self.rgbd_frame.iloc[idx, 0])
            with open(rgb_name, 'rb') as fRgb:
                rgb_image = Image.open(rgb_name).convert('RGB')
            
            depth_name = os.path.join(self.root_dir,
                                      self.rgbd_frame.iloc[idx, 1])
            with open(depth_name, 'rb') as fDepth:
                depth_image = Image.open(depth_name)
                
        # read input hdf5
        elif self.input_format == 'hdf5':
#            print('==> Input Format is hdf5')
            file_name = os.path.join(self.root_dir,
                                     self.rgbd_frame.iloc[idx, 0])
            rgb_h5, depth_h5 = self.load_h5(file_name)
            rgb_image = Image.fromarray(rgb_h5, mode='RGB')
            depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')
#            print(rgb_image.size)
#            plt.figure()
#            show_img(rgb_image)
#            plt.figure()
#            show_img(depth_image)
        else:
            print('error: the input format is not supported now!')
            return None
        
        _s = np.random.uniform(1.0, 1.5)
        degree = np.random.uniform(-5.0, 5.0)
        if self.split == 'train':
            tRgb = data_transform.Compose([data_transform.Crop(10, 1210, 130, 370),
                                           data_transform.Rotation(degree),
                                           transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4),
                                           transforms.CenterCrop((228, 912)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([data_transform.Crop(10, 1210, 130, 370),
                                             data_transform.Rotation(degree),
                                             transforms.CenterCrop((228, 912))])
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            if np.random.uniform()<0.5:
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)
            depth_image = depth_image.div(_s)           
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            

        elif self.split == 'val':
            tRgb = data_transform.Compose([data_transform.Crop(10, 1210, 130, 370),
                                           transforms.CenterCrop((228, 912)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([data_transform.Crop(10, 1210, 130, 370),
                                             transforms.CenterCrop((228, 912))])            
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            
        sample = {'rgbd': rgbd_image, 'depth': depth_image }
        
        return sample
    
    def createSparseDepthImage(self, depth_image, n_sample):
        random_mask = torch.zeros(1, depth_image.size(1), depth_image.size(2))
        n_pixels = depth_image.size(1) * depth_image.size(2)
        n_valid_pixels = torch.sum(depth_image>0.0001)
#        print('===> number of total pixels is: %d\n' % n_pixels)
#        print('===> number of total valid pixels is: %d\n' % n_valid_pixels)
        perc_sample = float(n_sample)/n_valid_pixels.float()
#        print(random_mask.type())
#        print(torch.ones_like(random_mask).type())
#        print(perc_sample)
        random_mask = torch.bernoulli((torch.ones_like(random_mask)*perc_sample))
        sparse_depth = torch.mul(depth_image, random_mask)
        return sparse_depth

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
    #    print (f.keys())
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)


def show_img(image):
    """Show image"""
    plt.imshow(image)


def test_load_h5():
    
    def load_h5(h5_filename):
        f = h5py.File(h5_filename, 'r')
    #    print (f.keys())
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)
    
    file_name = './data/kitti_hdf5/val/11/00466-R.h5'
    rgb_h5, depth_h5 = load_h5(file_name)   
    depth_h5 = depth_h5.astype('uint16')
    rgb_image = Image.fromarray(rgb_h5, mode='RGB')
    depth_image = Image.fromarray(depth_h5.astype('uint16'), mode='L')
#    cv2.imwrite('tmp/cv_save_kitti_depth.png', depth_h5)
    
#test_load_h5()

        
def test_imgread():
    # train preprocessing   
    kitti_dataset = KittiDataset(csv_file='data/kitti_hdf5/kitti_hdf5_train.csv',
                                       root_dir='.',
                                       split = 'train',
                                       n_sample = 500,
                                       input_format='hdf5')
#    kitti_dataset = KittiDataset(csv_file='data/nyudepth_v2/nyudepthv2_val.csv',
#                                       root_dir='.',
#                                       split = 'val',
#                                       n_sample = 500,
#                                       input_format='hdf5')    
    fig = plt.figure()
    for i in range(len(kitti_dataset)):
        sample = kitti_dataset[i]
        rgb = data_transform.ToPILImage()(sample['rgbd'][0:3,:,:])
        depth = data_transform.ToPILImage()(sample['depth'])
        sparse_depth = data_transform.ToPILImage()(sample['rgbd'][3,:,:].unsqueeze(0))
        depth_mask = data_transform.ToPILImage()(torch.sign(sample['depth']))
        sparse_depth_mask = data_transform.ToPILImage()(sample['rgbd'][3,:,:].unsqueeze(0).sign())
        print(sample['depth'])
        invalid_depth = torch.sum(sample['rgbd'][3,:,:].unsqueeze(0).sign() < 0)
        print(invalid_depth)
        plt.imsave("tmp/plt_save_kitit_depth.png", depth)
        depth.save(("tmp/pil_save_kitti_depth.png"))
        rgb.save("tmp/pil_save_kitti_rgb.png")
#        print(sample['depth'].size())
#        print(torch.sign(sample['sparse_depth']))
        ax = plt.subplot(5, 4, i + 1)
        ax.axis('off')
        show_img(rgb)
        ax = plt.subplot(5, 4, i + 5)
        ax.axis('off')
        show_img(depth)
        ax = plt.subplot(5, 4, i + 9)
        ax.axis('off')
        show_img(depth_mask)
        ax = plt.subplot(5, 4, i + 13)
        ax.axis('off')
        show_img(sparse_depth)
        ax = plt.subplot(5, 4, i + 17)
        ax.axis('off')
        show_img(sparse_depth_mask)
        if i == 3:
            plt.show()
            break
    
#test_imgread()
