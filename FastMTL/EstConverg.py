import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import itertools
import sys
sys.path.append('/home/lijunzhang/multibranch/')
from pathlib import Path
import random
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
from main.auto_models import MTSeqBackbone, MTSeqModel, ComputeBlock
from main.head import ASPPHeadNode
from main.trainer import Trainer
from main.algs_FMTL import complex_alignment, simple_alignment
from main.algs_EstCon import *

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/mnt/nfs/work1/huiguan/lijunzhang/multibranch/', help='project directory')
parser.add_argument('--dataroot', action='store', dest='dataroot', default='/mnt/nfs/work1/huiguan/lijunzhang/policymtl/data/', help='datasets directory')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='checkpoint/', help='checkpoints directory')
parser.add_argument('--exp_dir', action='store', dest='exp_dir', default='exp/', help='exp directory')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=32, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='mobilenet', help='backbone model')

parser.add_argument('--align', action='store', dest='align', default='complex', choices=['no','simple','complex'], help='align choices: no alignment, simple alignment, complex alignment')
parser.add_argument('--load_weight', action='store_true', help='whether load weight anchor')
parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='seed')

# Params for Est.
parser.add_argument('--total_iters', action='store', dest='total_iters', default=20000, type=int, help='total iterations')
parser.add_argument('--short_iters', action='store', dest='short_iters', default=500, type=int, help='short iterations for estimation')
parser.add_argument('--start', action='store', dest='start', default=10, type=int, help='start for estimation')
parser.add_argument('--step', action='store', dest='step', default=30, type=int, help='step for estimation')
parser.add_argument('--smooth_weight', action='store', dest='smooth_weight', default=0.0, type=float, help='smooth weight for loss list')
parser.add_argument('--target_ratio', action='store', dest='target_ratio', default=1.0, type=float, help='ratio for target loss')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')

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
    
    input_dim = (3,321,321)
elif args.data == 'Taskonomy':
    tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
    cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    
    dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataloader = DataLoader(dataset, batch_size=args.bz, shuffle=True)
    
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

################### Get Ind. Model Weights and Channel Alignment for Ind. Model ##############
if args.load_weight:
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

############################# Get Target Loss R0 ###############################
if args.data == 'NYUv2' and args.backbone == 'mobilenet':
    target = load_obj('./exp/','r0')
    print('Target: {}'.format(target), flush=True)
else:
    print('No Target Loss R0 Exists.')
    exit()
print('Finish Target Loss R0 Loading', flush=True)

####################### Enum Layouts and Estimate Convergency ######################
folder = '_'.join([str(args.load_weight), str(args.short_iters), str(args.smooth_weight)])+'/'
savepath = os.path.join(args.projectroot, args.ckpt_dir, args.data, args.exp_dir, folder)
Path(savepath).mkdir(parents=True, exist_ok=True)

# enum layout
layout_list = [] 
S0 = init_S(T, coarse_B) # initial state
L = Layout(T, coarse_B, S0) # initial layout
layout_list.append(L)
enum_layout_wo_rdt(L, layout_list)
print('Finish Layout Enumeration', flush=True)
print('='*60, flush=True)

total_iters, short_iters, start, step = args.total_iters, args.short_iters, args.start, args.step
smooth_weight, target_ratio = args.smooth_weight, args.target_ratio 
layout_est = []
layout_idx = -1
# For each layout
for L in layout_list:
    set_seed(args.seed)
    
    layout_idx += 1
    print('Layout {}'.format(layout_idx))
    layout = coarse_to_fined(L, fined_B, mapping)
    print('Fined Layout:', flush=True)
    print(layout, flush=True)
    
    mtl_model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num, verbose=False)
    if args.load_weight:
        # Step 1: create weight init state_dict
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
                    for block in model.backbone.mtl_blocks:
                        if task_set == block.task_set and block.layer_idx == layer_idx:
                            for ind_name, param in block.named_parameters():
                                mtl_init['.'.join([name, ind_name])] = param  
                            # for BN running mean and running var
                            for ind_name, param in block.named_buffers():
                                mtl_init['.'.join([name, ind_name])] = param

            # # Type 2: when the current block have merged operators, save mean weights for convs
            elif isinstance(module, Conv2dNode) and merge_flag: 
                task_convs = [] # store conv weights from task's ind. block
                for task in task_set:
                    # identify task-corresponding block in the well-trained ind. models 
                    for block in model.backbone.mtl_blocks:
                        if task in block.task_set and block.layer_idx == layer_idx:
                            task_module = block.compute_nodes[int(name.split('.')[-1])]  
                            temp_weight = task_module.basicOp.weight # no channel alignment or no align variable
                            if align_choice == 1 and task_module.out_ord is not None: # simple alignment
                                temp_weight = temp_weight[task_module.out_ord]
                            elif align_choice == 2: # complex alignment
                                if task_module.in_ord is not None:
                                    temp_weight = temp_weight[:,task_module.in_ord]
                                if task_module.out_ord is not None: 
                                    temp_weight = temp_weight[task_module.out_ord]
                            task_convs.append(temp_weight)
                weight_anchor = torch.mean(torch.stack(task_convs),dim=0)
                mtl_init[name+'.basicOp.weight'] = weight_anchor

            # Type 3: save heads' weights
            elif 'heads' in name and isinstance(module, ASPPHeadNode): 
                ind_head = model.heads[name.split('.')[-1]]
                for ind_name, param in ind_head.named_parameters():
                    mtl_init['.'.join([name, ind_name])] = param
                for ind_name, param in ind_head.named_buffers():
                    mtl_init['.'.join([name, ind_name])] = param
        mtl_model.load_state_dict(mtl_init,strict=False)
        print('Finish Weight Loading.', flush=True)
    
    # Step 2: Save short train loss list
    loss_lst = {task: [] for task in tasks}
    
    mtl_model = mtl_model.cuda()
    mtl_model.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, mtl_model.parameters()), lr=args.lr)
    trainIter = iter(trainDataloader)
    for i in range(short_iters):
        try:
            data = next(trainIter)
        except StopIteration:
            trainIter = iter(trainDataloader)
            data = next(trainIter)
            
        x = data['input'].cuda()
        optimizer.zero_grad()
        output = mtl_model(x)

        loss = 0
        for task in tasks:
            y = data[task].cuda()
            if task + '_mask' in data:
                tloss = criterionDict[task](output[task], y, data[task + '_mask'].cuda())
            else:
                tloss = criterionDict[task](output[task], y)
            loss_lst[task].append(tloss.item())
            loss += tloss
        loss.backward()
        optimizer.step()
    print('Finish Short Training.', flush=True)
    print('-'*80, flush=True)       
 
    # For each task
    alpha_lst = {task: [] for task in tasks}
    est_iter_lst ={task: [] for task in tasks}
    final_loss_lst = {task: [] for task in tasks}
    
    for task in tasks:
        print('Task {}:'.format(task), flush=True)
        sm_loss_lst = smooth(loss_lst[task], smooth_weight)
        alpha_num = ((short_iters - start)//step)//4
        
        for alpha_idx in range(alpha_num):
            print('\tAlpha Index {}:'.format(alpha_idx), flush=True)
            temp_start = start + (step*4)*alpha_idx
            # Step 3: Take smoothed loss samples from window slices
            loss_samples = window_loss_samples(sm_loss_lst, temp_start, step=step)
            print('\t\tLoss Samples: {}'.format(loss_samples), flush=True)
            if loss_samples == False:
                print('\t\tBad Loss Samples', flush=True)
                continue

            # Step 4: Compute convergence rate 
            alpha = compute_alpha2(loss_samples)
            alpha_lst[task].append(alpha)
            print('\t\tAlpha: {}'.format(alpha), flush=True)

            # Step 5,6: Estimate final loss after 20000 iters and iters to reach target loss
            n = (total_iters - temp_start)//step
            est_n, final_loss = est_final_loss2(loss_samples, n, alpha, target[task]*target_ratio)
            if est_n != -1:
                est_iter = start + est_n*step
            else:
                est_iter = est_n
            print('\t\tEst Iter: {}'.format(est_iter), flush=True)
            print('\t\tFinal Loss: {}'.format(final_loss), flush=True)
            est_iter_lst[task].append(est_iter)
            final_loss_lst[task].append(final_loss)
        print('-'*80, flush=True)
           
    results = {'alpha':alpha_lst, 'est_iter': est_iter_lst, 'est_loss': final_loss_lst}
    save_obj(results, savepath, 'layout_'+str(layout_idx))
    
    layout_est.append(results)
    print('='*80, flush=True)
    torch.cuda.empty_cache()

save_obj(layout_est, savepath, 'layout_est')



















