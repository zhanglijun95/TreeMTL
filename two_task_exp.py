import numpy as np
import random
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main.models import Deeplab_ASPP_Branch
from main.trainer import Trainer

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/mnt/nfs/work1/huiguan/lijunzhang/multibranch/', help='project directory')
parser.add_argument('--dataroot', action='store', dest='dataroot', default='/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/', help='datasets directory')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/', help='checkpoints directory')
parser.add_argument('--exp_dir', action='store', dest='exp_dir', default='exp/', help='save exp model directory')

parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='seed')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
parser.add_argument('--reload_ckpt', action='store', dest='reload_ckpt', default=None, help='reload model parameters file')
parser.add_argument('--coarse', action='store_true', help='whether use coarse branching points')

parser.add_argument('--branch', action='store', dest='branch', default=None, type=int, help='branching point')
parser.add_argument('--two_task', action='store', nargs='+', dest='two_task', default=[], help='two tasks to be leanred')

parser.add_argument('--val_iters', action='store', dest='val_iters', default=200, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=200, type=int, help='frequency of print')
parser.add_argument('--save_iters', action='store', dest='save_iters', default=200, type=int, help='frequency of model saving')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=4000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.5, type=float, help='rate of lr decay')

parser.add_argument('--total_iters', action='store', dest='total_iters', default=50000, type=int, help='total iterations')
parser.add_argument('--loss_lambda', action='store', nargs='+', dest='loss_lambda', default=[1,1], type=int, help='task loss weights')

args = parser.parse_args()
print(args, flush=True)
assert torch.cuda.is_available()

################################### Set Seed #####################################
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True) 

################################### Load Data #####################################
dataroot = os.path.join(args.dataroot, args.data)
two_task = args.two_task
if len(two_task) != 2:
    print('Wrong given tasks!')
    exit()

criterionDict = {}
metricDict = {}
clsNum = {}
if args.data == 'NYUv2':
    task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    
    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    trainDataloader = DataLoader(dataset, args.bz, shuffle=True)

    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, 8, shuffle=True)
    
    for task in two_task:
        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
        clsNum[task] = task_cls_num[task]
elif args.data == 'Taskonomy':
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)

    dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
    valDataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for task in two_task:
        criterionDict[task] = TaskonomyCriterions(task, dataroot)
        metricDict[task] = TaskonomyMetrics(task, dataroot)
        clsNum[task] = task_cls_num[task]
else:
    print('Wrong dataset!')
    exit()

################################ Generate Model ##################################
if args.coarse == True:
    print('Coarse Branching Points.', flush=True)
if args.backbone == 'resnet34':
    model = Deeplab_ASPP_Branch(args.branch, clsNum, args.coarse)
else:
    print('Wrong backbone!')
    exit()
model = model.cuda()
print('Finish Model Generation', flush=True)

############################# Generate Loss Lambda #################################
loss_lambda = {}
for content in zip(two_task, args.loss_lambda):
    loss_lambda[content[0]] = content[1]
    
################################# Train Model #####################################
print('Start Training', flush=True)
trainer = Trainer(model, two_task, trainDataloader, valDataloader, criterionDict, metricDict, 
            lr=args.lr, decay_lr_freq=args.decay_lr_freq, decay_lr_rate=args.decay_lr_rate, 
            print_iters=args.print_iters, val_iters=args.val_iters, save_iters=args.save_iters)
savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir)
Path(savepath).mkdir(parents=True, exist_ok=True)
trainer.train(args.total_iters, loss_lambda, savepath, reload=args.reload_ckpt)