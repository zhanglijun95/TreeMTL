import argparse
import numpy as np
import os
import itertools
import sys
import time
from pathlib import Path
from scipy import stats
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader

from framework.layer_node import Conv2dNode, InputNode
from main.layout import Layout
from main.algorithms import enum_layout_wo_rdt, init_S, coarse_to_fined
from main.algs_FMTL import simple_alignment, complex_alignment, dist_effort
from main.auto_models import MTSeqBackbone, MTSeqModel, ComputeBlock
from main.head import ASPPHeadNode
from main.trainer import Trainer

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/mnt/nfs/work1/huiguan/lijunzhang/multibranch/', help='project directory')
parser.add_argument('--dataroot', action='store', dest='dataroot', default='/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/', help='datasets directory')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/', help='checkpoints directory')
parser.add_argument('--writer_dir', action='store', dest='writer_dir', default='writer/', help='writer directory')
parser.add_argument('--exp_dir', action='store', dest='exp_dir', default='exp/', help='exp directory')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=16, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='mobilenet', help='backbone model')
parser.add_argument('--reload', action='store_true', help='whether reload ckpt')

parser.add_argument('--align', action='store', dest='align', default='no', choices=['no','simple','complex'], help='align choices: no alignment, simple alignment, complex alignment')
parser.add_argument('--mtl_load', action='store', dest='mtl_load', default='all', choices=['no', 'no_merged_only', 'merged_only', 'all'], help='multi-task model initial weights load choices: no init, only for non-merged blocks, only for merged blocks, and init for all')
parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=None, type=int, help='layout index')

parser.add_argument('--val_iters', action='store', dest='val_iters', default=200, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=50, type=int, help='frequency of print')
parser.add_argument('--save_iters', action='store', dest='save_iters', default=200, type=int, help='frequency of model saving')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=4000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.5, type=float, help='rate of lr decay')

parser.add_argument('--early_stop', action='store_true', help='whether early stop')
parser.add_argument('--stop_num', action='store', dest='stop_num', default=3, type=int, help='stop num for early stop')
parser.add_argument('--good_metric', action='store', dest='good_metric', default=10, type=int, help='at least how many good metrics every time, 10 for NYUv2')

parser.add_argument('--total_iters', action='store', dest='total_iters', default=20000, type=int, help='total iterations')
parser.add_argument('--loss_lambda', action='store', nargs='+', dest='loss_lambda', default=None, type=int, help='task loss weights')

parser.add_argument('--time', action='store_true', help='whether print time')
parser.add_argument('--order_verbose', action='store_true', help='whether print layout order')

args = parser.parse_args()
print(args, flush=True)
print('='*60, flush=True)
assert torch.cuda.is_available()


################################### Load Data #####################################
dataroot = os.path.join(args.dataroot, args.data)

criterionDict = {}
metricDict = {}

if args.data == 'NYUv2':
    tasks = ['segment_semantic', 'normal', 'depth_zbuffer']
    cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    
    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    trainDataloader = DataLoader(dataset, args.bz, shuffle=True)
    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, 8, shuffle=True) 
    
    input_dim = (3,321,321)
elif args.data == 'Taskonomy':
    tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
    cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)
    dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
    valDataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_dim = (3,224,224)
else:
    print('Wrong dataset!')
    exit()

T = len(tasks)
for task in tasks:
    print(task, flush=True)
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)

print('Finish Data Loading', flush=True)

########################## Params from Backbone #################################
if args.backbone == 'resnet34':
    prototxt = '../models/deeplab_resnet34_adashare.prototxt'  
    D = coarse_B = 5
    mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
    
elif args.backbone == 'mobilenet':
    prototxt = '../models/mobilenetv2.prototxt'
    D = coarse_B = 5
    mapping = {0:[0,1,2,3,4,5,6], 1:[7,8,9,10,11,12,13,14,15,16,17], 2:[18,19,20,21,22], 
           3:[23,24,25,26,27,28,29,30], 4:[31], 5:[32]}
    
else:
    print('Wrong backbone!')
    exit()

# prepare params (number of blocks and feature dim) automatically 
with torch.no_grad():
    backbone = MTSeqBackbone(prototxt)
    fined_B = len(backbone.basic_blocks)
    feature_dim = backbone(torch.rand(1,3,224,224)).shape[1]
    
if max(mapping[max(mapping)]) != fined_B:
    print('Wrong mapping for the given backbone model because of inconsistent number of blocks.')
    exit()
    
print('Finish Preparing Backbone Params', flush=True)

############################# Load Ind. Model Weights ###############################
ckptroot = os.path.join(args.projectroot, args.ckpt_dir, args.data, 'ind', args.backbone)
ind_name = '_'.join(tasks) + '.model'
weight_PATH = os.path.join(ckptroot, ind_name)

# ind. layout
S = []
for i in range(fined_B):
    S.append([set([x]) for x in range(T)])
layout = Layout(T, fined_B, S) 

# ind. model
with torch.no_grad():
    model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num, verbose=False)
    # load ind. model weights
    model.load_state_dict(torch.load(weight_PATH)['state_dict'])

print('Finish Ind. Model Weights Loading', flush=True)

######################### Channel Alignment for Ind. Model ########################
start = time.time()
align_choice = args.align

if align_choice == 'simple':
    print('Simple Channel Alignment', flush=True)
    simple_alignment(model, tasks)
elif align_choice == 'complex':
    print('Complex Channel Alignment', flush=True)
    complex_alignment(model, tasks)
elif align_choice == 'no':
    pass

print('Finish Ind. Model Weights Channel Alignment', flush=True)

####################### Enum Layouts and Compute Merging Cost ######################
# enum layout
layout_list = [] 
S0 = init_S(T, coarse_B) # initial state
L = Layout(T, coarse_B, S0) # initial layout
layout_list.append(L)
enum_layout_wo_rdt(L, layout_list)
print('Finish Layout Enumeration', flush=True)
    
if args.layout_idx == None:
    print('='*60, flush=True)
    print('Compute Merging Cost', flush=True)
    for L in layout_list:
        layout = coarse_to_fined(L, fined_B, mapping)
        if args.order_verbose:
            print('Fined Layout:', flush=True)
            print(layout, flush=True)

        merge_num = 0
        sum_efforts = 0
        for b in range(layout.B):
            sets = layout.state[b]
            for task_set in sets:
                # if a task set has more than 1 element, merging happens
                if len(task_set) > 1:
                    merge_num += len(task_set) - 1
                    task_convs = [] # store conv weights in each block corresponding to each task

                    for task in task_set:
                        # identify task-corresponding block in the well-trained ind. models 
                        temp = []
                        for block in model.backbone.mtl_blocks:
                            if task in block.task_set and block.layer_idx == b:
                                for module in block.compute_nodes:
                                    if isinstance(module, Conv2dNode):
                                        temp_weight = module.basicOp.weight.detach().numpy() # no channel alignment or no align variable

                                        if align_choice == 'simple' and module.out_ord is not None: # simple alignment
                                            temp_weight = temp_weight[module.out_ord]
                                        elif align_choice == 'complex': # complex alignment
                                            if module.in_ord is not None:
                                                temp_weight = temp_weight[:,module.in_ord]
                                            if module.out_ord is not None: 
                                                temp_weight = temp_weight[module.out_ord]

                                        temp.append(temp_weight)
                        task_convs.append(temp)

                    # compute effort for each conv
                    for conv_idx in range(len(task_convs[0])):
                        conv_weights = []
                        for task_idx in range(len(task_set)):
                            conv_weights.append(task_convs[task_idx][conv_idx])
                        conv_weights = np.array(conv_weights)
                        weight_anchor = np.mean(conv_weights, axis=0) 
                        sum_efforts += dist_effort(conv_weights, weight_anchor)

        if merge_num == 0:
            L.effort = 0
        else:
            L.effort = sum_efforts/merge_num
        if args.order_verbose:
            print('Effort: {:7f}'.format(L.effort), flush=True)
            print('-'*60, flush=True)

    print('Finish Computing Merging Cost', flush=True)
    print('='*60, flush=True)

    # sort
    layout_order = sorted(range(len(layout_list)), key=lambda k: layout_list[k],reverse=False)
    layout_idx = layout_order[1]
else:
    layout_idx = args.layout_idx
print('Selected Layout Idx:{}'.format(layout_idx), flush=True)
end = time.time()
if args.time:
    print('Time for Layout Sort: {:4f}'.format(end-start), flush=True)

######################### Generate MTL Model and Weights ##########################
layout = coarse_to_fined(layout_list[layout_idx], fined_B, mapping)
mtl_model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num)

start = time.time()
# create weight init state_dict
if args.mtl_load == 'no':
    pass
else:
    print('Generate MTL Weights', flush=True)
    mtl_init = OrderedDict()
    for name, module in mtl_model.named_modules():
        if isinstance(module, ComputeBlock):
            task_set = module.task_set
            layer_idx = module.layer_idx
            if len(task_set) > 1:
                merge_flag = True
            else:
                # Type 1: save the whole block weights from the corresponding ind. model when no merging
                merge_flag = False
                if args.mtl_load in ['no_merged_only', 'all']:
                    for block in model.backbone.mtl_blocks:
                        if task_set == block.task_set and block.layer_idx == layer_idx:
                            for ind_name, param in block.named_parameters():
                                mtl_init['.'.join([name, ind_name])] = param  
                            # for BN running mean and running var
                            for ind_name, param in block.named_buffers():
                                mtl_init['.'.join([name, ind_name])] = param

        # # Type 2: when the current block have merged operators, save mean weights for convs
        elif isinstance(module, Conv2dNode) and merge_flag and args.mtl_load in ['merged_only', 'all']:
            task_convs = [] # store conv weights from task's ind. block
            for task in task_set:
                # identify task-corresponding block in the well-trained ind. models 
                for block in model.backbone.mtl_blocks:
                    if task in block.task_set and block.layer_idx == layer_idx:
                        task_module = block.compute_nodes[int(name.split('.')[-1])]  
                        temp_weight = task_module.basicOp.weight # no channel alignment or no align variable
                        if align_choice == 'simple' and task_module.out_ord is not None: # simple alignment
                            temp_weight = temp_weight[task_module.out_ord]
                        elif align_choice == 'complex': # complex alignment
                            if task_module.in_ord is not None:
                                temp_weight = temp_weight[:,task_module.in_ord]
                            if task_module.out_ord is not None: 
                                temp_weight = temp_weight[task_module.out_ord]
                        task_convs.append(temp_weight)
            weight_anchor = torch.mean(torch.stack(task_convs),dim=0)
            mtl_init[name+'.basicOp.weight'] = weight_anchor

        # Type 3: save heads' weights
        elif 'heads' in name and isinstance(module, ASPPHeadNode) and args.mtl_load in ['no_merged_only', 'all']: 
            ind_head = model.heads[name.split('.')[-1]]
            for ind_name, param in ind_head.named_parameters():
                mtl_init['.'.join([name, ind_name])] = param
            for ind_name, param in ind_head.named_buffers():
                mtl_init['.'.join([name, ind_name])] = param

    mtl_model.load_state_dict(mtl_init,strict=False)
end = time.time()
if args.time:
    print('Time for MTL Weight Gen: {:4f}'.format(end-start), flush=True)
    
print('Finish MTL Model Generation and Weights Init', flush=True)

############################# Generate Loss Lambda #################################
if args.loss_lambda is not None:
    loss_lambda = {}
    for content in zip(tasks, args.loss_lambda):
        loss_lambda[content[0]] = content[1]
else:
    loss_lambda = {task: 1 for task in tasks}
    
################################# Train Model #####################################
print('Start Training', flush=True)   
    
folder = '_'.join([str(layout_idx),align_choice,args.mtl_load])+'/'
savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir, folder)
writerpath = os.path.join(args.projectroot, args.writer_dir, args.data, args.exp_dir, folder)
Path(savepath).mkdir(parents=True, exist_ok=True)
Path(writerpath).mkdir(parents=True, exist_ok=True)
    
mtl_model = mtl_model.cuda()
trainer = Trainer(mtl_model, tasks, trainDataloader, valDataloader, criterionDict, metricDict, 
            lr=args.lr, decay_lr_freq=args.decay_lr_freq, decay_lr_rate=args.decay_lr_rate, 
            print_iters=args.print_iters, val_iters=args.val_iters, save_iters=args.save_iters,
            early_stop=args.early_stop, stop=args.stop_num, good_metric=args.good_metric)
trainer.train(args.total_iters, loss_lambda, savepath, writerPath=writerpath)    
    
print('Finish Training', flush=True)         