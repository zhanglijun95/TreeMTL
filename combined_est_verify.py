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
from main.algorithms import enum_layout_wo_rdt, init_S, coarse_to_fined
from main.auto_models import MTSeqBackbone, MTSeqModel
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
parser.add_argument('--reload', action='store_true', help='whether reload ckpt')

parser.add_argument('--verify', action='store_true', help='whether verifying high-order layouts or training first-order layouts')
parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=None, type=int, help='layout index')
parser.add_argument('--branch', action='store', dest='branch', default=None, type=int, help='branching point')
parser.add_argument('--two_task', action='store', nargs='+', dest='two_task', default=[], help='two tasks to be leanred')

parser.add_argument('--val_iters', action='store', dest='val_iters', default=200, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=200, type=int, help='frequency of print')
parser.add_argument('--save_iters', action='store', dest='save_iters', default=200, type=int, help='frequency of model saving')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=4000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.5, type=float, help='rate of lr decay')

parser.add_argument('--total_iters', action='store', dest='total_iters', default=50000, type=int, help='total iterations')
parser.add_argument('--loss_lambda', action='store', nargs='+', dest='loss_lambda', default=None, type=int, help='task loss weights')

args = parser.parse_args()
print(args, flush=True)
print('='*60, flush=True)
assert torch.cuda.is_available()

################################### Set Seed #####################################
seed = args.seed
if seed != 0:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) 
else:
    print('Default Seed 0 -> No Seed', flush=True)

################################### Exp Type ######################################
if args.verify:
    print('Verify High-Order Layouts', flush=True)
    exp = 'verify'
else:
    print('Train First-Order Layouts', flush=True)
    exp = '2task'
print('='*60, flush=True)

################################### Load Data #####################################
dataroot = os.path.join(args.dataroot, args.data)
two_task = args.two_task
if len(two_task) != 2 and exp == '2task':
    print('Wrong given tasks!')
    exit()

criterionDict = {}
metricDict = {}
cls_num = {}
if args.data == 'NYUv2':
    all_tasks = ['segment_semantic','normal','depth_zbuffer']
    task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    
    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    trainDataloader = DataLoader(dataset, args.bz, shuffle=True)

    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, 8, shuffle=True)
    
    if exp == '2task':
        tasks = two_task
    elif exp == 'verify':
        tasks = all_tasks
    
    for task in tasks:
        print(task, flush=True)
        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
        cls_num[task] = task_cls_num[task]
elif args.data == 'Taskonomy':
    all_tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
    task_cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)

    dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
    valDataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    if exp == '2task':
        tasks = two_task
    elif exp == 'verify':
        tasks = all_tasks
        
    for task in tasks:
        criterionDict[task] = TaskonomyCriterions(task, dataroot)
        metricDict[task] = TaskonomyMetrics(task, dataroot)
        cls_num[task] = task_cls_num[task]
else:
    print('Wrong dataset!')
    exit()

print('Finish Data Loading', flush=True)

########################## Params from Backbone #################################
if args.backbone == 'resnet34':
    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    backbone_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/resnet34_backbone.model')
    if args.data == 'NYUv2':
        heads_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/resnet34_NYUv2_heads.model')
    elif args.data == 'Taskonomy':
        heads_init_path = None
        
    coarse_B = 5
    mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
    
elif args.backbone == 'mobilenet':
    prototxt = 'models/mobilenetv2.prototxt'
    backbone_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/mobilenetv2_backbone.model')
    if args.data == 'NYUv2':
        heads_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/mobilenetv2_NYUv2_heads.model')
    elif args.data == 'Taskonomy':
        backbone_init_path = None
        heads_init_path = None
        
#     coarse_B = 9
#     mapping = {0:[0], 1:[1,2], 2:[3,4,5,6], 3:[7,8,9,10,11], 4:[12,13,14,15,16,17], 5:[18,19,20,21,22], 
#               6:[23,24,25,26,27], 7:[28,29,30], 8:[31], 9:[32]} 
#     coarse_B = 6
#     mapping = {0:[0], 1:[1,2,3,4,5,6], 2:[7,8,9,10,11], 3:[12,13,14,15,16,17], 4:[18,19,20,21,22,23,24,25,26,27], 5:[28,29,30,31], 6:[32]} 
    coarse_B = 5
    mapping = {0:[0,1,2,3,4,5,6], 1:[7,8,9,10,11,12,13,14,15,16,17], 2:[18,19,20,21,22], 
           3:[23,24,25,26,27,28,29,30], 4:[31], 5:[32]} 
    
elif args.backbone == 'mobilenetS':
    prototxt = 'models/mobilenetv2_shorter.prototxt'
    backbone_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/mobilenetv2_shorter_backbone.model')
    if args.data == 'NYUv2':
        heads_init_path = os.path.join(args.projectroot, args.ckpt_dir, 'init/mobilenetv2_shorter_NYUv2_heads.model')
    elif args.data == 'Taskonomy':
        backbone_init_path = None
        heads_init_path = None
    
    coarse_B = 8
    mapping = {0:[0], 1:[1,2], 2:[3,4,5,6], 3:[7,8,9,10,11], 4:[12,13,14,15,16,17], 5:[18,19,20,21,22], 
           6:[23,24,25,26,27], 7:[28,29,30], 8:[31]} 
        
#     coarse_B = 4
#     mapping = {0:[0,1,2,3,4,5,6], 1:[7,8,9,10,11,12,13,14,15,16,17], 2:[18,19,20,21,22], 
#            3:[23,24,25,26,27,28,29,30], 4:[31]} 
    
else:
    print('Wrong backbone!')
    exit()

# prepare params (number of blocks and feature dim) automatically 
with torch.cuda.device(0):
    backbone = MTSeqBackbone(prototxt)
    fined_B = len(backbone.basic_blocks)
    feature_dim = backbone(torch.rand(1,3,224,224)).shape[1]
    
if max(mapping[max(mapping)]) != fined_B:
    print('Wrong mapping for the given backbone model because of inconsistent number of blocks.')
    exit()
    
print('Finish Preparing Backbone Params', flush=True)

####################### Transfer Layouts OR Branch Points##############################
if exp == '2task':
    branch = args.branch
    print('Coarse Branch Point:', flush=True)
    print(branch, flush=True)

    branch = mapping[branch][0]
    print('Fined Branch Point:', flush=True)
    print(branch, flush=True)
    
    print('Finish Branch Point Transfer', flush=True)
    
elif exp == 'verify':
    T = len(tasks)
    
    layout_list = [] 
    S0 = init_S(T, coarse_B) # initial state
    L = Layout(T, coarse_B, S0) # initial layout
    layout_list.append(L)

    enum_layout_wo_rdt(L, layout_list)
    
    layout = layout_list[args.layout_idx]
    print('Coarse Layout:', flush=True)
    print(layout, flush=True)

    layout = coarse_to_fined(layout, fined_B, mapping)
    print('Fined Layout:', flush=True)
    print(layout, flush=True)
    
    print('Finish Layout Emueration and Selection', flush=True)
print('='*60, flush=True)

################################ Generate Model ##################################
if exp == '2task':
    model = MTSeqModel(prototxt, branch=branch, fined_B=fined_B, feature_dim=feature_dim, cls_num=cls_num, 
                 backbone_init=backbone_init_path, heads_init=heads_init_path)
elif exp == 'verify':
    model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num, 
                 backbone_init=backbone_init_path, heads_init=heads_init_path)
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

if exp == '2task':
    savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir)
elif exp == 'verify':
    savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir, str(args.layout_idx)+'/')
Path(savepath).mkdir(parents=True, exist_ok=True)

if args.reload is False:
    trainer.train(args.total_iters, loss_lambda, savepath)
else:
    if exp == '2task':
        reload_ckpt = '_'.join(tasks) + '_b' + str(branch) + '.model'
    elif exp == 'verify':
        reload_ckpt = '_'.join(tasks) + '.model'
    trainer.train(args.total_iters, loss_lambda, savepath, reload_ckpt)
