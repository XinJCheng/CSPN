"""
Created on Fri Feb  2 19:16:42 2018

@ author:  Xinjing Cheng
@ email : chengxinjing@baidu.com
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import sys
import argparse
from torch.autograd import Variable
import utils
import loss as my_loss

parser = argparse.ArgumentParser(description='PyTorch Sparse To Dense Evaluation')

# net parameters
parser.add_argument('--n_sample', default=200, type=int, help='sampled sparse point number')
parser.add_argument('--data_set', default='nyudepth', type=str, help='train dataset')

# optimizer parameters
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')

# network parameters
parser.add_argument('--cspn_step', default=24, type=int, help='steps of propagation')
parser.add_argument('--cspn_norm_type', default='8sum', type=str, help='norm type of cspn')

# batch size
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size for eval')

#data directory
parser.add_argument('--save_dir', default='result/base_line', type=str, help='result save directory')
parser.add_argument('--best_model_dir', default='result/base_line', type=str, help='best model load directory')
parser.add_argument('--train_list', default='datalist/nyudepth_hdf5_train.csv', type=str, help='train data lists')
parser.add_argument('--eval_list', default='datalist/nyudepth_hdf5_val.csv', type=str, help='eval data list')
parser.add_argument('--model', default='base_model', type=str, help='model for net')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained resnet model')

args = parser.parse_args()

sys.path.append("./models")
import update_model
if args.model == 'cspn_unet':
    if args.data_set=='nyudepth':
        print("==> evaluating model with cspn and unet on nyudepth")
        import torch_resnet_cspn_nyu as model
    elif args.data_set =='kitti':
        print("==> evaluating model with cspn and unet on kitti")
        import torch_resnet_cspn_kitti as model
else:
    import torch_resnet as model

use_cuda = torch.cuda.is_available()

# global variable
best_rmse = sys.maxsize  # best test rmse
cspn_config = {'step': args.cspn_step, 'norm_type': args.cspn_norm_type}

# Data
print('==> Preparing data..')
assert args.data_set in ['nyudepth', 'kitti']
if args.data_set=='nyudepth':
    import eval_nyu_dataset_loader as dataset_loader
    valset = dataset_loader.NyuDepthDataset(csv_file=args.eval_list,
                                            root_dir='.',
                                            split = 'val',
                                            n_sample = args.n_sample,
                                            input_format='hdf5')
elif args.data_set =='kitti':
    import eval_kitti_dataset_loader as dataset_loader
    valset = dataset_loader.KittiDataset(csv_file=args.eval_list,
                                         root_dir='.',
                                         split = 'val',
                                         n_sample = args.n_sample,
                                         input_format='hdf5')
else:
    print("==> input unknow dataset..")

valloader = torch.utils.data.DataLoader(valset,
                                        batch_size=args.batch_size_eval,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last=False)
# Model
print('==> Building model..')

if args.data_set == 'nyudepth':
    net = model.resnet50(cspn_config=cspn_config)
elif args.data_set == 'kitti':
    net = model.resnet18(cspn_config=cspn_config)
else:
    print("==> input unknow dataset..")

if True:
    # Load best model checkpoint.
    print('==> Resuming from best model..')
    best_model_path = os.path.join(args.best_model_dir, 'best_model.pth')
    assert os.path.isdir(args.best_model_dir), 'Error: no checkpoint directory found!'
    best_model_dict = torch.load(best_model_path)
    best_model_dict = update_model.remove_moudle(best_model_dict)
    net.load_state_dict(update_model.update_model(net, best_model_dict))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = my_loss.Wighted_L1_Loss().cuda()

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      nesterov=args.nesterov,
                      dampening=args.dampening)

# evaluation
def val(epoch):
    net.eval()
    total_step_val = 0
    error_sum_val = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                     'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                     'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0}
    error_avg = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                 'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                 'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0}
    for batch_idx, sample in enumerate(valloader):
        [inputs, targets, raw_rgb] = [sample['rgbd'] , sample['depth'], sample['raw_rgb']]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        targets = targets.data.cpu()
        outputs = outputs.data.cpu()
        loss = loss.data.cpu()

        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)

        total_step_val += args.batch_size_eval
        error_avg = utils.avg_error(error_sum_val,
                                    error_result,
                                    total_step_val,
                                    args.batch_size_eval)
        utils.print_error('eval_result: step(average)',
                          epoch, batch_idx,
                          loss, error_result, error_avg)
        utils.save_eval_img(args.data_set, args.best_model_dir, batch_idx,
                            inputs.data.cpu(), raw_rgb, targets, outputs)

    utils.print_single_error(epoch, batch_idx, loss, error_avg)

def eval_error():
    val(0)

eval_error()

