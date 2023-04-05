import sys
sys.path.append('../')
import numpy as np
import random
import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main.layout import Layout
from main.algorithms import enum_layout_wo_rdt, init_S, coarse_to_fined
from main.efficientnet import EffNetV2_FC
from main.trainer_domainnet import DomainNetTrainer
from data.domainnet_dataloader import DomainNet, MultiDomainSampler

parser = argparse.ArgumentParser()
parser.add_argument('--projectroot', action='store', dest='projectroot', default='/work/lijunzhang_umass_edu/data/multibranch/', help='project directory')
parser.add_argument('--exp_dir', action='store', dest='exp_dir', default='exp/', help='save exp model directory')

parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='seed')
parser.add_argument('--data', action='store', dest='data', default='DomainNet', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='batch_size', default=32, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='effnetv2', help='backbone model')
parser.add_argument('--reload', action='store_true', help='whether reload ckpt')

parser.add_argument('--verify', action='store_true', help='whether verifying high-order layouts or training first-order layouts')
parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=None, type=int, help='layout index')
parser.add_argument('--branch', action='store', dest='branch', default=None, type=int, help='branching point')
parser.add_argument('--two_task', action='store', nargs='+', dest='two_task', default=[], help='two tasks to be leanred')

parser.add_argument('--total_iters', action='store', dest='total_iters', default=20000, type=int, help='total iterations')
parser.add_argument('--val_iters', action='store', dest='val_iters', default=400, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=100, type=int, help='frequency of print')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=2000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.3, type=float, help='rate of lr decay')

args = parser.parse_args()
print(args, flush=True)
print('='*60, flush=True)
assert torch.cuda.is_available()

################################### Exp Type ######################################
if args.verify:
    print('Verify High-Order Layouts', flush=True)
    exp = 'verify'
else:
    print('Train First-Order Layouts', flush=True)
    exp = '2task'
print('='*60, flush=True)

################################### Load Data #####################################
dataroot = os.path.join(args.projectroot, 'data/', args.data)
if exp == '2task' and len(args.two_task) != 2:
    print('Wrong given tasks!')
    exit()
    
if args.data == 'DomainNet':
    if exp == '2task':
        tasks = args.two_task
    else:
        tasks = ['real', 'painting', 'quickdraw', 'clipart', 'infograph', 'sketch']
    cls_num = {task: 345 for task in tasks}
    
    dataloader = {}
    with open(os.path.join(dataroot, 'DomainNet.json'), 'r') as f:
        split_domain_info = json.load(f)
    for mode in ['train','val']:
        random_shuffle = True if mode == 'train' else False
        sampler = MultiDomainSampler(split_domain_info, mode, args.batch_size, tasks, random_shuffle)
        dataset = DomainNet(dataroot, mode)
        dataloader[mode] = DataLoader(dataset, batch_sampler=sampler)
        print('size of %s dataloader: ' % mode, len(dataloader[mode]))
else:
    print('Wrong dataset!')
    exit()

print('Finish Data Loading', flush=True)

########################## Params from Backbone #################################
if args.backbone == 'effnetv2':
    coarse_B = 4
    fined_B = 42
    mapping = {0: [0, 1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10],
               2: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
               3: [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], 4: [42]}
else:
    print('Wrong backbone!')
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

    enum_layout_wo_rdt(L, layout_list, args.layout_idx)
    
    layout = layout_list[args.layout_idx]
    print('Coarse Layout:', flush=True)
    print(layout, flush=True)

    layout = coarse_to_fined(layout, fined_B, mapping)
    print('Fined Layout:', flush=True)
    print(layout, flush=True)
    
    print('Finish Layout Emueration and Selection', flush=True)
print('='*60, flush=True)

################################ Generate Model ################################## 
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if exp == '2task':
    model = EffNetV2_FC(tasks=tasks, branch=branch, cls_num=cls_num)
elif exp == 'verify':
    model = EffNetV2_FC(tasks=tasks, layout=layout, cls_num=cls_num)
model = model.cuda()
print('Finish Model Generation', flush=True)

################################# Train Model #####################################
print('Start Training', flush=True)
trainer = DomainNetTrainer(model, tasks, dataloader['train'], dataloader['val'], batch_size=args.batch_size, 
                           lr=args.lr, decay_lr_freq=args.decay_lr_freq, decay_lr_rate=args.decay_lr_rate, 
                           print_iters=args.print_iters, val_iters=args.val_iters)
if exp == '2task':
    savepath = os.path.join(args.projectroot, 'checkpoint/', args.data, args.exp_dir)
elif exp == 'verify':
    savepath = os.path.join(args.projectroot, 'checkpoint/', args.data, args.exp_dir, str(args.layout_idx)+'/')
Path(savepath).mkdir(parents=True, exist_ok=True)

if args.reload is False:
    trainer.train(args.total_iters, savePath=savepath)
else:
    if exp == '2task':
        reload_ckpt = '_'.join(tasks) + '_b' + str(branch) + '.model'
    elif exp == 'verify':
        reload_ckpt = '_'.join(tasks) + '.model'
    trainer.train(args.total_iters, savePath=savepath, reload=reload_ckpt)