import numpy as np
import random
import torch
import pickle


def save_obj(obj, path, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def window_loss_samples(loss_lst, start, step=None, indices=None):
    if (step != None and start + step * 4 > len(loss_lst)) or (indices !=None and len(indices) != 4):
        print('Wrong Window Slices for Loss Samples!')
        return False
    if step != None:
        samples = [np.mean(loss_lst[(start+step*i):(start+step*(i+1))]) for i in range(4)]
    elif indices !=None:
        samples = [np.mean(loss_lst[start:indices[i]]) if i == 0 else np.mean(loss_lst[indices[i-1]:indices[i]]) for i in range(4)]
    else:
        print('No Slices or Step designed for Loss Samples!')
        return False
    if judge_loss_samples(samples):
        return samples
    else:
        print('\t\tLoss Samples: {}'.format(samples), flush=True)
        return False

def judge_loss_samples(samples):
    diff_prev = 1000
    for i in range(1, len(samples)):
        prev = samples[i-1]
        cur = samples[i]
        diff = prev - cur
        if prev < cur or diff > diff_prev:
            return False
        else:
            diff_prev = diff
    return True

def good_loss_samples(loss_lst, indices, tol=10):
    if len(indices) != 4:
        print('Wrong Indices for Loss Samples!')
        return False
    # sample 0
    samples = [loss_lst[indices[0]]]
    # sample 1,2,3
    diff_prev = 1000
    for idx in range(1,len(indices)):
        prev = samples[idx-1]
        cur = loss_lst[indices[idx]]
        diff = prev - cur
        if cur < prev and diff < diff_prev:
            samples.append(cur)
            diff_prev = diff
        else:
            for i in range(indices[idx]-tol,indices[idx]+tol):
                prev = samples[idx-1]
                cur = loss_lst[i]
                diff = prev - cur
                if cur < prev and diff < diff_prev:
                    samples.append(cur)
                    diff_prev = diff
                    break
                else:
                    return False
    return samples

def compute_alpha(loss_samples):
    return np.log(loss_samples[2]/loss_samples[1])/ np.log(loss_samples[1]/loss_samples[0])

def compute_alpha2(loss_samples):
    return np.log(np.abs((loss_samples[3]-loss_samples[2])/(loss_samples[2]-loss_samples[1])))/ \
            np.log(np.abs((loss_samples[2]-loss_samples[1])/(loss_samples[1]-loss_samples[0])))

def est_final_loss(loss_samples, n, alpha):
    x0, x1 = np.log(loss_samples[0]), np.log(loss_samples[1])
    for i in range(2, n+1):
        x2 = alpha * (x1-x0) + x1
        x0 = x1
        x1 = x2
    return np.exp(x2)

def est_final_loss2(loss_samples, n, alpha, target):
    x0, x1 = np.log(loss_samples[0]-loss_samples[1]), np.log(loss_samples[1]-loss_samples[2])
    temp = loss_samples[2]
    est_n = -1
    flag = True
    
    for i in range(2, n+1):
        x2 = alpha * (x1-x0) + x1
        if np.isinf(x2):
            break
        else:
            temp -= np.exp(x2)
        if temp < target and flag:
            est_n = i
            flag = False
        x0 = x1
        x1 = x2
    return est_n, temp
