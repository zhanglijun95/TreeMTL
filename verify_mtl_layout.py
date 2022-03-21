import numpy as np
import random
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main.layout import Layout
from main.algorithms import enum_layout, enum_layout_wo_rdt, init_S, coarse_to_fined
from main.models import Deeplab_ASPP_Layout
from main.trainer import Trainer

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/mnt/nfs/work1/huiguan/lijunzhang/multibranch/', help='project directory')
parser.add_argument('--dataroot', action='store', dest='dataroot', default='/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/', help='datasets directory')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/', help='checkpoints directory')
parser.add_argument('--exp_dir', action='store', dest='exp_dir', default='verify/', help='save exp model directory')

parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='seed')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
parser.add_argument('--reload', action='store_true', help='whether reload ckpt')
parser.add_argument('--wo_rdt', action='store_true', help='whether remove redundancy')
parser.add_argument('--coarse', action='store_true', help='whether use coarse branching points')

parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=None, type=int, help='layout index')

parser.add_argument('--val_iters', action='store', dest='val_iters', default=200, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=50, type=int, help='frequency of print')
parser.add_argument('--save_iters', action='store', dest='save_iters', default=200, type=int, help='frequency of model saving')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=4000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.5, type=float, help='rate of lr decay')

parser.add_argument('--total_iters', action='store', dest='total_iters', default=50000, type=int, help='total iterations')
parser.add_argument('--loss_lambda', action='store', nargs='+', dest='loss_lambda', default=None, type=int, help='task loss weights')

args = parser.parse_args()
print(args, flush=True)
assert torch.cuda.is_available()

################################### Set Seed #####################################
seed = args.seed
if seed != 0:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
else:
    print('Default Seed 0 -> No Seed', flush=True)

################################### Load Data #####################################
dataroot = os.path.join(args.dataroot, args.data)

criterionDict = {}
metricDict = {}
clsNum = {}
if args.data == 'NYUv2':
    tasks = ['segment_semantic','normal','depth_zbuffer']
    task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    trainDataloader = DataLoader(dataset, args.bz, shuffle=True)

    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, args.bz, shuffle=True)

    for task in tasks:
        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
elif args.data == 'Taskonomy':
    tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)

    dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
    valDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)
    
    for task in tasks:
        criterionDict[task] = TaskonomyCriterions(task, dataroot)
        metricDict[task] = TaskonomyMetrics(task, dataroot)
else:
    print('Wrong dataset!')
    exit()
    
########################## Enum all layouts for given T and B #################################
T = len(tasks) 
if args.backbone == 'resnet34':
    if args.coarse == True:
        print('Coarse Branching Points.', flush=True)
        B = 5
        fined_B = 17
        mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
    else:
        B = 17 
else:
    print('Wrong backbone!')
    exit()
    
layout_list = [] 
S0 = init_S(T, B) # initial state
L = Layout(T, B, S0) # initial layout
layout_list.append(L)

if args.wo_rdt == False:
    enum_layout(L, layout_list)
else:
    print('Layouts w/o redundancy.', flush=True)
    enum_layout_wo_rdt(L, layout_list)
print('Finish Layout Emueration', flush=True)
    
################################ Generate Model ##################################
layout = layout_list[args.layout_idx]
print('Layout:', flush=True)
print(layout, flush=True)

if args.coarse == True:
    layout = coarse_to_fined(layout, fined_B, mapping)
    print('Fined Layout:', flush=True)
    print(layout, flush=True)
    
if args.backbone == 'resnet34':
    model = Deeplab_ASPP_Layout(layout, task_cls_num)
model = model.cuda()
print('Finish Model Generation', flush=True)

############################# Generate Loss Lambda #################################
loss_lambda = {}
if args.loss_lambda is not None:
    for content in zip(tasks, args.loss_lambda):
        loss_lambda[content[0]] = content[1]
else:
    for task in tasks:
        loss_lambda[task] = 1
    
################################# Train Model #####################################
print('Start Training', flush=True)
trainer = Trainer(model, tasks, trainDataloader, valDataloader, criterionDict, metricDict, 
            lr=args.lr, decay_lr_freq=args.decay_lr_freq, decay_lr_rate=args.decay_lr_rate, 
            print_iters=args.print_iters, val_iters=args.val_iters, save_iters=args.save_iters)
savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir, str(args.layout_idx)+'/')
Path(savepath).mkdir(parents=True, exist_ok=True)
if args.reload is False:
    trainer.train(args.total_iters, loss_lambda, savepath)
else:
    reload_ckpt = '_'.join(tasks) + '.model'
    trainer.train(args.total_iters, loss_lambda, savepath, reload_ckpt)
