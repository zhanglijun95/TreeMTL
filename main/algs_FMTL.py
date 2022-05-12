import numpy as np
from scipy.optimize import linear_sum_assignment
from statistics import mean
import pickle

from main.auto_models import ComputeBlock
from framework.layer_node import Conv2dNode, InputNode

def dist_effort(weight_list, weight_anchor):
    effort = 0
    for weight in weight_list:
        effort += np.linalg.norm(weight - weight_anchor) / weight.size # naive normalization
    return effort

def simple_alignment(model, tasks, conv_num=-1, verbose=False):
    # Step 1: store tasks' conv modules 
    T = len(tasks)
    task_anchor = tasks[0]
    
    taskConvList = {task: [] for task in tasks}
    count = 0
    for name, module in model.named_modules():
        if count >= conv_num * T and conv_num != -1:
            break
        if isinstance(module, ComputeBlock):
            if len(module.task_set) == 1:
                task_idx = list(module.task_set)[0]
            else:
                print('Wrong independent models with merged branching blocks.')
                sys.exit()
        elif isinstance(module, Conv2dNode):
            taskConvList[tasks[task_idx]].append(module)
            count += 1
    
    # Step 2: align conv weights to task_anchor's in channels
    pre_dist = []
    post_dist = []
    for task in tasks:
        if task == task_anchor:
            continue

        for taskConv in zip(taskConvList[task_anchor],taskConvList[task]):
            # extract weights before alignment
            chn_weights = []
            for i in range(2):
                op = taskConv[i].basicOp
                temp = op.weight.view(op.out_channels, -1) #(out_channels, in_channels*kernel_size), no alignment from prev conv
                chn_weights.append(temp.detach().numpy())

            # compute distance
            dist1 = np.sum(np.linalg.norm(chn_weights[0] - chn_weights[1], axis=1))
            pre_dist.append(dist1)

            # channel alignment
            cost_mat = np.sum(np.power(chn_weights[0], 2),axis=1, keepdims=True) + \
                       np.sum(np.power(chn_weights[1].T, 2),axis=0, keepdims=True) - 2 * (chn_weights[0]@chn_weights[1].T)
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            # save col_ind to ConvNode.out_ord of the second task
            taskConv[1].out_ord = col_ind

            # compute distance
            dist2 = np.sum(np.linalg.norm(chn_weights[0] - chn_weights[1][col_ind,:], axis=1))
            post_dist.append(dist2)
    # print
    if verbose:
        print('pre dist:{}'.format(mean(pre_dist)))
        print('post dist:{}'.format(mean(post_dist)))        
    return

def complex_alignment(model, tasks, conv_num=-1, verbose=False):
    def type_in_list(typ, lst):
        for val in lst:
            if isinstance(val, typ):
                return True
        return False

    def argmin(lst):
        return min(range(len(lst)), key=lst.__getitem__)

    def argmax(lst):
        return max(range(len(lst)), key=lst.__getitem__)

    def father_conv(model, module): 
        fathers = module.fatherNodeList
        if type_in_list(InputNode, fathers):
            # except the first node
            return False
        else:
            # find the father nodes containing Conv2dNode recursively
            while not type_in_list(Conv2dNode, fathers):
                fidx = argmax([x.nodeIdx for x in fathers])
                fathers = fathers[fidx].fatherNodeList
            for f in fathers:
                if isinstance(f, Conv2dNode):
                    return f

    # Step 1: store tasks' conv modules
    T = len(tasks)
    task_anchor = tasks[0]
    
    taskConvList = {task: [] for task in tasks}
    count = 0
    for name, module in model.named_modules():
        if count >= conv_num * T and conv_num != -1:
            break
        if isinstance(module, ComputeBlock):
            if len(module.task_set) == 1:
                task_idx = list(module.task_set)[0]
            else:
                print('Wrong independent models with merged branching blocks.')
                sys.exit()
        elif isinstance(module, Conv2dNode):
            taskConvList[tasks[task_idx]].append(module)
            count += 1

    # Step 2: align conv weights to task_anchor's in channels
    pre_dist = []
    post_dist = []
    for task in tasks:
        if task == task_anchor:
            continue

        for taskConv in zip(taskConvList[task_anchor],taskConvList[task]):
            f = father_conv(model,taskConv[1])
            if f:
                # Corner Case: for GroupConv, set out_ord = father.out_ord, then stop
                if taskConv[1].basicOp.groups > 1:
                    taskConv[1].out_ord = f.out_ord
                    continue
                # Step 2-1: for general Conv, set in_ord=father.out_ord
                else:
                    taskConv[1].in_ord = f.out_ord

            # Step 2-2: extract weights before alignment
            chn_weights = []
            for i in range(2):
                op = taskConv[i].basicOp
                if taskConv[i].in_ord is not None:
                    temp = op.weight[:,taskConv[i].in_ord]
                else:
                    temp = op.weight
                temp = temp.view(op.out_channels, -1) #(out_channels, in_channels*kernel_size)
                chn_weights.append(temp.detach().numpy())

            # compute distance (not good version)
            dist1 = np.sum(np.linalg.norm(chn_weights[0] - \
                      taskConv[1].basicOp.weight.view(op.out_channels, -1).detach().numpy(), axis=1))
            pre_dist.append(dist1)

            # Step 2-3: channel alignment
            cost_mat = np.sum(np.power(chn_weights[0], 2),axis=1, keepdims=True) + \
                       np.sum(np.power(chn_weights[1].T, 2),axis=0, keepdims=True) - 2 * (chn_weights[0]@chn_weights[1].T)
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            # save col_ind to ConvNode.out_ord of the second task
            taskConv[1].out_ord = col_ind

            # compute distance
            dist2 = np.sum(np.linalg.norm(chn_weights[0] - chn_weights[1][col_ind,:], axis=1))
            post_dist.append(dist2)
    # print
    if verbose:
        print('pre dist:{}'.format(mean(pre_dist)))
        print('post dist:{}'.format(mean(post_dist)))
    return
