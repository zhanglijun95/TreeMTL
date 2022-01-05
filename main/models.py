import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import ASPPHeadNode

affine_par = True

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=full_padding, dilation=dilation, bias=False)

# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__() 
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y
    
# Add residual projection
class ResidualBlock(nn.Module):
    def __init__(self, block, ds):
        super(ResidualBlock, self).__init__() 
        self.block = block
        self.ds = ds
        
    def forward(self, x):
        residual = self.ds(x) if self.ds is not None else x
        x = F.relu(residual + self.block(x))
        return x
    
class Deeplab_ResNet_Backbone_Branch(nn.Module):
    def __init__(self, block, layers, branch=None, task_num=2):
        super(Deeplab_ResNet_Backbone_Branch, self).__init__()
        
        self.inplanes = 64
        self.branch = branch
        self.task_num = task_num

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.shared_blocks, self.separate_blocks = [], []
        self.layer_config = layers
        
        branch_cnt = 0
        seed = self._make_seed()
        self.__add_to_share_or_separate(branch_cnt, seed)
        branch_cnt += 1
        
        for segment, num_blocks in enumerate(self.layer_config):
            filt_size, num_blocks, stride, dilation = filt_sizes[segment],layers[segment],strides[segment],dilations[segment]
            for b_idx in range(num_blocks):
                blocklayer = self._make_blocklayer(b_idx, block, filt_size, stride=stride, dilation=dilation)
                self.__add_to_share_or_separate(branch_cnt, blocklayer)
                branch_cnt += 1

        self.shared_blocks = nn.ModuleList(self.shared_blocks)
        self.separate_blocks = nn.ModuleList(self.separate_blocks) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_seed(self):
        seed = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                             nn.BatchNorm2d(64, affine=affine_par),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)) 
        return seed
    
    def _make_downsample(self, block, inplanes, planes, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
        return downsample
    
    def _make_blocklayer(self, block_idx, block, planes, stride=1, dilation=1):
        ds = None
        if block_idx == 0:
            basic_block = block(self.inplanes, planes, stride, dilation=dilation)
            ds = self._make_downsample(block, self.inplanes, planes, stride=stride, dilation=dilation)
            self.inplanes = planes * block.expansion
        else:
            basic_block = block(self.inplanes, planes, dilation=dilation)
            
        blocklayer = ResidualBlock(basic_block, ds)
        return blocklayer
    
    def __add_to_share_or_separate(self, branch_cnt, block):
        if self.branch is None or branch_cnt < self.branch:
            self.shared_blocks.append(block)
        else:
            multiple_blocks = []
            for i in range(self.task_num):
                multiple_blocks.append(copy.deepcopy(block))
            self.separate_blocks.append(nn.ModuleList(multiple_blocks))
        return

    def forward(self, x): 
        for block in self.shared_blocks:
            x = block(x)
            
        output = [x] * self.task_num
        for multiple_blocks in self.separate_blocks:
            for i in range(self.task_num):
                output[i] = multiple_blocks[i](output[i])
        return output
    
class Deeplab_ASPP_Branch(nn.Module):
    def __init__(self, branch, cls_num, coarse=False):
        super(Deeplab_ASPP_Branch, self).__init__()
        self.branch = branch
        if coarse:
            mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
            b = mapping[branch][0]
        else:
            b = branch
        print('Real Branching Point: '+str(b), flush=True)
        self.backbone = Deeplab_ResNet_Backbone_Branch(BasicBlock, [3, 4, 6, 3], b, len(cls_num))
        self.heads = nn.ModuleDict()
        for task in cls_num:
            self.heads[task] = ASPPHeadNode(512, cls_num[task])
        
    def forward(self, x):
        features = self.backbone(x)
        output = {}
        idx = 0
        for task in self.heads:
            output[task] = self.heads[task](features[idx])
            idx += 1
        return output
    
    

####################### Transfrom Layout to MT-Model ####################
class LayoutBlock(nn.Module):
    def __init__(self, residual_block, task_set, layer_idx):
        super(LayoutBlock, self).__init__() 
        self.residual_block = residual_block
        self.task_set = task_set
        self.layer_idx = layer_idx
        
        # set outside 
        self.parent_block = [] 
        self.output = None
        
    def forward(self, x=None):
        if len(self.parent_block) == 0:
            if x is not None:
                return self.residual_block(x)
            else:
                print('Wrong Input Data!')
                return
        else:
            return self.residual_block(self.parent_block[0].output)
        
class Deeplab_ResNet_Backbone_Layout(nn.Module):
    def __init__(self, block, layers, layout):
        super(Deeplab_ResNet_Backbone_Layout, self).__init__()
        
        # Step 1: Sanity Check - The number of blocks in Deeplab_Resnet34 should be 17
        if sum(layers) + 1 != layout.B:
            print('Given layout cannot construct multi-task model of Deeplab_Resnet34 because of the incompatiable number of blocks.')
            return
        
        # Step 2: Initiate Backbone Blocks - Modules in self.blocks are not used in forward
        self.inplanes = 64
        self.layout = layout
    
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks = []
        self.layer_config = layers
        
        branch_cnt = 0
        seed = self._make_seed()
        self.__add_to_blocks(seed)
        branch_cnt += 1
        
        for segment, num_blocks in enumerate(self.layer_config):
            filt_size, num_blocks, stride, dilation = filt_sizes[segment],layers[segment],strides[segment],dilations[segment]
            for b_idx in range(num_blocks):
                blocklayer = self._make_blocklayer(b_idx, block, filt_size, stride=stride, dilation=dilation)
                self.__add_to_blocks(blocklayer)
#         self.blocks = nn.ModuleList(self.blocks)
        
        # Step 3: Copy Blocks to Construct Multi-Task Model
        self.mtl_blocks = []
        for layer_idx in range(self.layout.B):
            basic_block = self.blocks[layer_idx]
            for task_set in self.layout.state[layer_idx]:
                layoutblock = LayoutBlock(copy.deepcopy(basic_block), task_set, layer_idx)
                # set parent block except the first layer
                if layer_idx != 0:
                    for block in self.mtl_blocks:
                        if block.layer_idx == layer_idx - 1 and task_set.issubset(block.task_set):
                            layoutblock.parent_block.append(block)
                            break
                self.mtl_blocks.append(layoutblock)
        self.mtl_blocks = nn.ModuleList(self.mtl_blocks)
        
#         Step 4: Initiate Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # Functions for Backbone Blocks
    def _make_seed(self):
        seed = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                             nn.BatchNorm2d(64, affine=affine_par),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)) 
        return seed
    
    def _make_downsample(self, block, inplanes, planes, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
        return downsample
    
    def _make_blocklayer(self, block_idx, block, planes, stride=1, dilation=1):
        ds = None
        if block_idx == 0:
            basic_block = block(self.inplanes, planes, stride, dilation=dilation)
            ds = self._make_downsample(block, self.inplanes, planes, stride=stride, dilation=dilation)
            self.inplanes = planes * block.expansion
        else:
            basic_block = block(self.inplanes, planes, dilation=dilation)
            
        blocklayer = ResidualBlock(basic_block, ds)
        return blocklayer
    
    def __add_to_blocks(self, block):
        self.blocks.append(block)
        return   
    
    def forward(self, x): 
        features = [0] * self.layout.T
        for block in self.mtl_blocks:
            block.output = block(x)
            if block.layer_idx == self.layout.B - 1:
                for task in block.task_set:
                    features[task] = block.output
        return features
    
class Deeplab_ASPP_Layout(nn.Module):
    def __init__(self, layout, cls_num=None):
        super(Deeplab_ASPP_Layout, self).__init__()
        self.layout = layout
        self.backbone = Deeplab_ResNet_Backbone_Layout(BasicBlock, [3, 4, 6, 3], layout)
        self.heads = nn.ModuleDict()
        if cls_num is not None:
            for task in cls_num:
                self.heads[task] = ASPPHeadNode(512, cls_num[task])
        
    def forward(self, x):
        features = self.backbone(x)
        if len(self.heads) == 0:
            return features
        else:
            output = {}
            idx = 0
            for task in self.heads:
                output[task] = self.heads[task](features[idx])
                idx += 1
            return output