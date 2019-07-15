# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:27:01 2018

@author: norbot
"""

import torch
import math
import os
import data_transform
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)

def evaluate_error(gt_depth, pred_depth):
    # for numerical stability
    depth_mask = gt_depth>0.0001
    batch_size = gt_depth.size(0)
    error = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
             'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
             'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,
             }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth   = gt_depth[depth_mask]
    n_valid_element = _gt_depth.size(0)

    if n_valid_element > 0:
        diff_mat = torch.abs(_gt_depth-_pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)
        error['MSE'] = torch.sum(torch.pow(diff_mat, 2))/n_valid_element
        error['RMSE'] = math.sqrt(error['MSE'])
        error['MAE'] = torch.sum(diff_mat)/n_valid_element
        error['ABS_REL'] = torch.sum(rel_mat)/n_valid_element
        y_over_z = torch.div(_gt_depth, _pred_depth)
        z_over_y = torch.div(_pred_depth, _gt_depth)
        max_ratio = max_of_two(y_over_z, z_over_y)
        error['DELTA1.02'] = torch.sum(max_ratio < 1.02).numpy()/float(n_valid_element)
        error['DELTA1.05'] = torch.sum(max_ratio < 1.05).numpy()/float(n_valid_element)
        error['DELTA1.10'] = torch.sum(max_ratio < 1.10).numpy()/float(n_valid_element)
        error['DELTA1.25'] = torch.sum(max_ratio < 1.25).numpy()/float(n_valid_element)
        error['DELTA1.25^2'] = torch.sum(max_ratio < 1.25**2).numpy()/float(n_valid_element)
        error['DELTA1.25^3'] = torch.sum(max_ratio < 1.25**3).numpy()/float(n_valid_element)
    return error

# avg the error
def avg_error(error_sum, error_step, total_step, batch_size):
    error_avg = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                 'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                 'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}
    for item, value in error_step.items():
        error_sum[item] += error_step[item] * batch_size
        error_avg[item] = error_sum[item]/float(total_step)
    return error_avg


# print error
def print_error(split, epoch, step, loss, error, error_avg, print_out=False):
    format_str = ('%s ===>\n\
                  Epoch: %d, step: %d, loss=%.4f\n\
                  MSE=%.4f(%.4f)\tRMSE=%.4f(%.4f)\tMAE=%.4f(%.4f)\tABS_REL=%.4f(%.4f)\n\
                  DELTA1.02=%.4f(%.4f)\tDELTA1.05=%.4f(%.4f)\tDELTA1.10=%.4f(%.4f)\n\
                  DELTA1.25=%.4f(%.4f)\tDELTA1.25^2=%.4f(%.4f)\tDELTA1.25^3=%.4f(%.4f)\n')
    error_str = format_str % (split, epoch, step, loss,\
                         error['MSE'], error_avg['MSE'], error['RMSE'], error_avg['RMSE'],\
                         error['MAE'], error_avg['MAE'], error['ABS_REL'], error_avg['ABS_REL'],\
                         error['DELTA1.02'], error_avg['DELTA1.02'], \
                         error['DELTA1.05'], error_avg['DELTA1.05'], \
                         error['DELTA1.10'], error_avg['DELTA1.10'], \
                         error['DELTA1.25'], error_avg['DELTA1.25'], \
                         error['DELTA1.25^2'], error_avg['DELTA1.25^2'], \
                         error['DELTA1.25^3'], error_avg['DELTA1.25^3'])
    if print_out:
        print(error_str)
    return error_str


def print_single_error(epoch, step, loss, error):
    format_str = ('%s ===>\n\
                  Epoch: %d, step: %d, loss=%.4f\n\
                  MSE=%.4f\tRMSE=%.4f\tMAE=%.4f\tABS_REL=%.4f\n\
                  DELTA1.02=%.4f\tDELTA1.05=%.4f\tDELTA1.10=%.4f\n\
                  DELTA1.25=%.4f\tDELTA1.25^2=%.4f\tDELTA1.25^3=%.4f\n')
    print (format_str % ('eval_avg_error', epoch, step, loss,\
                         error['MSE'], error['RMSE'], error['MAE'],  error['ABS_REL'], \
                         error['DELTA1.02'], error['DELTA1.05'], error['DELTA1.10'], \
                         error['DELTA1.25'], error['DELTA1.25^2'], error['DELTA1.25^3']))

# update_best_model
def updata_best_model(error_avg, best_RMSE):
    if error_avg['RMSE'] < best_RMSE:
        return True
    else:
        return False

# log best_model
def log_file_folder_make(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, 0o777)

    train_log_file = os.path.join(save_dir, 'log_train.txt')
    train_fd = open(train_log_file, 'w')
    train_fd.write('epoch\t bestModel\t MSE\t RMSE\t MAE\t \
                   DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                   DELTA1.25^2\t DELTA1.25^3\t ABS_REL\n')
    train_fd.close()

    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    eval_fd = open(eval_log_file, 'w')
    eval_fd.write('epoch\t bestModel\t MSE\t RMSE\t MAE\t \
                  DELTA1.02\t DELTA1.05\t DELTA1.10\t \
                  DELTA1.25\t DELTA1.25^2\t DELTA1.25^3\t ABS_REL\n')
    eval_fd.close()

def log_result(save_dir, error_avg, epoch, lr, best_model, split):
    format_str = ('%.4f\t %.4f\t\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n')
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    if split == 'train':
        train_fd = open(train_log_file, 'a')
        train_fd.write(format_str%(epoch, best_model, error_avg['MSE'], error_avg['RMSE'],\
                                   error_avg['MAE'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],\
                                   error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],\
                                   error_avg['DELTA1.25^3'], error_avg['ABS_REL']))
        train_fd.close()
    elif split == 'eval':
        eval_fd = open(eval_log_file, 'a')
        eval_fd.write(format_str%(epoch, best_model, error_avg['MSE'], error_avg['RMSE'],\
                                  error_avg['MAE'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],\
                                  error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],\
                                  error_avg['DELTA1.25^3'], error_avg['ABS_REL']))
        eval_fd.close()

# log best_model
def log_file_folder_make_lr(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, 0o777)
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    train_fd = open(train_log_file, 'w')
    train_fd.write('epoch\t lr\t bestModel\t MSE\t RMSE\t MAE\t \
                   DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                   DELTA1.25^2\t DELTA1.25^3\t ABS_REL\n')
    train_fd.close()

    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    eval_fd = open(eval_log_file, 'w')
    eval_fd.write('epoch\t lr\t bestModel\t MSE\t RMSE\t MAE\t \
                  DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                  DELTA1.25^2\t DELTA1.25^3\t ABS_REL\n')
    eval_fd.close()

def log_result_lr(save_dir, error_avg, epoch, lr, best_model, split):
    format_str = ('%.4f\t %.4f\t %.4f\t\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n')
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    if split == 'train':
        train_fd = open(train_log_file, 'a')
        train_fd.write(format_str%(epoch, lr, best_model, error_avg['MSE'], error_avg['RMSE'],\
                                   error_avg['MAE'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],\
                                   error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],\
                                   error_avg['DELTA1.25^3'], error_avg['ABS_REL']))
        train_fd.close()
    elif split == 'eval':
        eval_fd = open(eval_log_file, 'a')
        eval_fd.write(format_str%(epoch, lr, best_model, error_avg['MSE'], error_avg['RMSE'],\
                                  error_avg['MAE'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],\
                                  error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],\
                                  error_avg['DELTA1.25^3'], error_avg['ABS_REL']))
        eval_fd.close()


def un_normalize(tensor):
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    for t, m, s in zip(tensor, img_mean, img_std):
        t.mul_(s).add_(m)
    return tensor

def save_eval_img(data_set, model_dir, index, input_rgbd, input_rgb, gt_depth, pred_depth):
    img_save_folder = os.path.join(model_dir, 'eval_result')
    if not os.path.isdir(img_save_folder):
        os.makedirs(img_save_folder, 0o777)

    save_name_rgb = os.path.join(img_save_folder, "%05d_input.png" % (index))
    save_name_gt = os.path.join(img_save_folder, "%05d_gt.png" % (index))
    save_name_pred = os.path.join(img_save_folder, "%05d_pred.png" % (index))
    save_name_sparse_point = os.path.join(img_save_folder, "%05d_sparse_point.png" % (index))
    save_name_sparse_mask = os.path.join(img_save_folder, "%05d_sparse_mask.png" % (index))
    save_rgb = transforms.ToPILImage()(torch.squeeze(input_rgb, 0))
    save_gt = None
    save_pred = None
    if data_set == 'kitti':
        save_sparse_point = data_transform.ToPILImage()(input_rgbd[:,3,:,:])
        save_sparse_mask = data_transform.ToPILImage()(input_rgbd[:,3,:,:].sign())
        save_gt = data_transform.ToPILImage()(torch.squeeze(gt_depth*1.0, 0))
        save_pred = data_transform.ToPILImage()(torch.squeeze(pred_depth*1.0, 0))
        plt.imsave(save_name_rgb, save_rgb)
        plt.imsave(save_name_gt, save_gt)
        plt.imsave(save_name_pred, save_pred)

    elif data_set == 'nyudepth':
        save_gt = data_transform.ToPILImage()(torch.squeeze(gt_depth*25.5, 0))
        save_pred = data_transform.ToPILImage()(torch.squeeze(pred_depth*25.5, 0))
        save_rgb.save(save_name_rgb)
        save_gt.save(save_name_gt)
        save_pred.save(save_name_pred)

def test_eval_error():
    gt_depth = torch.abs(torch.randn(1,3,4))
    pred_depth = torch.abs(torch.randn(1,3,4))
    eval_result = evaluate_error(gt_depth, pred_depth)
    for item, value in eval_result.items():
        print(('%s\' value is: %f') %(item, eval_result[item]))
