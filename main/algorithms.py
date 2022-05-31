import numpy as np
import antropy as ant
import copy
import itertools
import random
from statistics import mean
from main.layout import Layout

def enumerator(T, B):
    layout_list = [] 
    
    S0 = init_S(T, B) # initial state
    L = Layout(T, B, S0) # initial layout
    layout_list.append(L)
    
    enum_layout_wo_rdt(L, layout_list)
    return layout_list

def enum_layout(L, layout_list):
    # Main Function:
    ##  Recursively enumerate all possible layouts
    
    # Exit Case: If the number of cut applied to L has already reached T-1, no more cut can be applied
    if L.num_cut >= L.T-1:
        return
    # Enumerate all possible layout cuts on L
    for i in range(L.lowest_avail_cut, L.B): # Cut for every avaiable block
        for task_set in L.state[i]: # Cut for each task set in ith block
            if len(task_set) == 1: # If the task set has only 1 element, it cannot be cut anymore
                continue
            subsets_list = divide_set(task_set) # Find all possible 2 subsets
            for subsets in subsets_list: # For any 2 subsets
                L_prime = apply_cut(L, i, task_set, subsets) # Construct new layout based on the cut
                layout_list.append(L_prime)
                enum_layout(L_prime, layout_list)
    return

def remove_redundancy(layout_list):
    # Main Function:
    ##  Remove redundant layout
    ##  May merge with enum_layout
    new_layout_list = []
    for L in layout_list:
        if L not in new_layout_list:
            new_layout_list.append(L)
    return new_layout_list

def enum_layout_wo_rdt(L, layout_list):
    # Main Function:
    ##  Recursively enumerate all possible layouts
    
    # Exit Case: If the number of cut applied to L has already reached T-1, no more cut can be applied
    if L.num_cut >= L.T-1:
        return
    # Enumerate all possible layout cuts on L
    for i in range(L.lowest_avail_cut, L.B): # Cut for every avaiable block
        for task_set in L.state[i]: # Cut for each task set in ith block
            if len(task_set) == 1: # If the task set has only 1 element, it cannot be cut anymore
                continue
            subsets_list = divide_set(task_set) # Find all possible 2 subsets
            for subsets in subsets_list: # For any 2 subsets
                L_prime = apply_cut(L, i, task_set, subsets) # Construct new layout based on the cut
                if L_prime not in layout_list:
                    layout_list.append(L_prime)
                    enum_layout_wo_rdt(L_prime, layout_list)
    return

def coarse_to_fined(L, fined_B, mapping):
    # Main Function:
    ## Convert layouts with coarse branching points to fine-grained ones
    
    S = []
    for idx in range(L.B):
        for times in mapping[idx]:
            S.append(L.state[idx])
    new_L = Layout(L.T, fined_B, S)
    return new_L

def metric_inference(L, two_task_metrics=None):
    # Main Function:
    ## Figure out the subtrees consisting the given layout
    ## Compute the estimate metric for each task based on the subtrees' results
    
    S = L.state
    subtree_dict = {}
    for t1 in range(L.T):
        subtree = []
        for t2 in range(L.T):
            if t1 == t2:
                continue
            
            branch = L.B # No branch - All share
            for i in range(L.B): # For eac block
                share_flag = False
                for task_set in S[i]: # For each task set in ith block
                    # There exists a task set has both t1 and t2 -> t1 and t2 share in ith block
                    if t1 in task_set and t2 in task_set: 
                        share_flag = True
                        break
                if share_flag is False:
                    branch = i
                    break
            subtree.append([t1, t2, branch])
        subtree_dict[t1] = subtree
        if two_task_metrics is not None:
            L.metric_list[t1] = get_metric(two_task_metrics, subtree)
    return subtree_dict
        
def prob_inference(L, two_task_prob):
    # Main Function:
    ## Figure out the subtrees consisting the given layout
    ## Compute the joint probability for the layout based on the subtrees' probability
    
    S = L.state
    tasks = [i for i in range(L.T)]
    subtree = []
    for t1,t2 in itertools.combinations(tasks, 2):
        branch = L.B # No branch - All share
        for i in range(L.B): # For eac block
            share_flag = False
            for task_set in S[i]: # For each task set in ith block
                # There exists a task set has both t1 and t2 -> t1 and t2 share in ith block
                if t1 in task_set and t2 in task_set: 
                    share_flag = True
                    break
            if share_flag is False:
                branch = i
                break
        subtree.append([t1, t2, branch])
    print('subtree: {}'.format(subtree))
    L.prob = joint_prob(two_task_prob, subtree)

# Helper functions
def init_S(T, B):
    # Function:
    ##  Construct the first state for given task and branch number
    ##  S is a list of list of task sets, e.g., [[{1,2,3}], [{1,2,3}], [{1,2},{3}]]
    S = []
    for i in range(B):
        task_sets = []
        task_sets.append(set([x for x in range(T)]))
        S.append(task_sets)
    return S

def divide_set(task_set):
    # Function:
    ##   Construct all possible 2 subsets for given task_set based on the binary theory
    ##   Remove duplicates and empty subset
    ##  If len(task_set) = n, len(subsets_list) = 2^(n-1) - 1
    subsets_list = []
    for pattern in itertools.product([True,False],repeat=len(task_set)):
        if pattern[0] is False and sum(pattern) != 0: # Exclude the duplicates and empty subset
            l1 = set([x[1] for x in zip(pattern,task_set) if x[0]])
            l2 = task_set - l1
            subsets = [l1, l2]
            subsets_list.append(subsets)    
    return subsets_list

def apply_cut(L, i, task_set, subsets):
    # Function:
    ## Apply a given cut on a layout to generate a new one
    L_new = copy.deepcopy(L)
    for j in range(i, L.B):
        L_new.state[j].remove(task_set) # Remove the cutted task set
        L_new.state[j] += subsets # Add the 2 new subsets of task set
    L_new.num_cut += 1 # Update the number of cuts
    L_new.lowest_avail_cut = i # Update the lowest available cut position
    return L_new

def reorg_two_task_results(two_task_pd, T, B):
    tasks = [i for i in range(T)]
    two_task_metrics = {}
    for two_set in itertools.combinations(tasks, 2):
        two_task_metrics[two_set] = []
        for b in range(0, B+1): 
            metric1 = two_task_pd[str(two_set)+'-0'][b]
            metric2 = two_task_pd[str(two_set)+'-1'][b]
            two_task_metrics[two_set].append([metric1, metric2])
    return two_task_metrics

def compute_weights(two_task_pd, T):
    fluc = {t: [] for t in range(T)}
    for two_set in itertools.combinations([i for i in range(T)], 2):
        for idx in range(2):
            metric = two_task_pd[str(two_set)+'-'+str(idx)].tolist()[1:]
            vol = ant.svd_entropy(metric, normalize=True)
            fluc[two_set[idx]].append(vol)

    score_weights = []
    for key in fluc:
        score_weights.append(1-np.mean(fluc[key]))
    score_weights_norm = [float(i)/sum(score_weights) for i in score_weights]
    return score_weights_norm

def get_metric(two_task_metrics, subtree):
    # Function:
    ## Return the average metric of target task based on the subtree' two_task_metrics results
    metric_list = []
    for two_task_branch in subtree:
        branch = two_task_branch[2]
        
        two_set = (two_task_branch[0], two_task_branch[1])
        if two_set in two_task_metrics:
            metric_list.append(two_task_metrics[two_set][branch][0])

        two_set = (two_task_branch[1], two_task_branch[0])
        if two_set in two_task_metrics:
            metric_list.append(two_task_metrics[two_set][branch][1])
    return mean(metric_list)

def joint_prob(two_task_prob, subtree):
    # Function:
    ## Return the joint probability based on the subtree' probability   
    prob_list = []
    for two_task_branch in subtree:
        branch = two_task_branch[2]
        
        two_set = (two_task_branch[0], two_task_branch[1])
        if two_set in two_task_prob:
            prob_list.append(two_task_prob[two_set][branch])

        two_set = (two_task_branch[1], two_task_branch[0])
        if two_set in two_task_prob:
            prob_list.append(two_task_prob[two_set][branch])
    return np.prod(prob_list)