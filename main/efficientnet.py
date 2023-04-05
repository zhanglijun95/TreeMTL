"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""
import copy
import math
from sys import exit
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from main.layout import Layout

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                # nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                nn.Conv2d(oup, _make_divisible(inp // reduction, 8), kernel_size=1),
                nn.SiLU(inplace=True),
                # nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Conv2d(_make_divisible(inp // reduction, 8), oup, kernel_size=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup, eps=0.001),
        nn.SiLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, eps=0.001),
        nn.SiLU(inplace=True)
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se, first_two, sd_prob):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=0.001),
                nn.SiLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=0.001),
                nn.SiLU(inplace=True),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=0.001),
            )
        elif first_two:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=0.001),
                nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=0.001),
                nn.SiLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=0.001),
            )
        self.stochastic_depth = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        result = self.conv(x)
        if self.identity:
            result = self.stochastic_depth(result)
            result += x
        return result

################################################ Original EffNetV2 ######################################
class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        
        # building inverted residual blocks
        block = MBConv
        total_stage_blocks = sum(cfg[2] for cfg in self.cfgs)
        stage_block_id = 0
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                sd_prob = 0.2 * float(stage_block_id) / total_stage_blocks
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se, (t==1), sd_prob))
                input_channel = output_channel
                stage_block_id += 1
        
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        layers.append(conv_1x1_bn(input_channel, output_channel))
        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(output_channel, num_classes),)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

############################################ EffNetV2 for Branched Models ######################################
class EffNetV2_Backbone_Branch(nn.Module):
    def __init__(self, cfgs, branch=None, task_num=2, width_mult=1.):
        super(EffNetV2_Backbone_Branch, self).__init__()
        self.cfgs = cfgs
        self.branch = branch
        self.task_num = task_num
        self.shared_blocks, self.separate_blocks = [], []

        # building first layer
        branch_cnt = 0
        input_channel = _make_divisible(24 * width_mult, 8)
        firstblock = conv_3x3_bn(3, input_channel, 2)
        self.__add_to_share_or_separate(branch_cnt, firstblock)
        branch_cnt += 1
        
        # building inverted residual blocks
        block = MBConv
        total_stage_blocks = sum(cfg[2] for cfg in self.cfgs)
        stage_block_id = 0
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                sd_prob = 0.2 * float(stage_block_id) / total_stage_blocks
                blocklayer = block(input_channel, output_channel, s if i == 0 else 1, t, use_se, (t==1), sd_prob)
                self.__add_to_share_or_separate(branch_cnt, blocklayer)
                input_channel = output_channel
                stage_block_id += 1
                branch_cnt += 1
        
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        lastblock = conv_1x1_bn(input_channel, output_channel)
        self.__add_to_share_or_separate(branch_cnt, lastblock)
        
        self.shared_blocks = nn.ModuleList(self.shared_blocks)
        self.separate_blocks = nn.ModuleList(self.separate_blocks)

        self.__initialize_weights()
        
    def __add_to_share_or_separate(self, branch_cnt, block):
        if self.branch is None or branch_cnt < self.branch:
            self.shared_blocks.append(block)
        else:
            multiple_blocks = []
            for i in range(self.task_num):
                multiple_blocks.append(copy.deepcopy(block))
            self.separate_blocks.append(nn.ModuleList(multiple_blocks))
        return

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    
    def forward(self, x, task_idx):
        for block in self.shared_blocks:
            x = block(x)
        for multiple_blocks in self.separate_blocks:
            x = multiple_blocks[task_idx](x)
        return x

class EffNetV2_FC_Branch(nn.Module):
    def __init__(self, tasks, branch, cls_num, coarse=True):
        super(EffNetV2_FC_Branch, self).__init__()
        self.branch = branch
        self.tasks = tasks
        
        cfgs = [
        # t, c, n, s, SE for EffNet_s
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
        ]
        self.backbone = EffNetV2_Backbone_Branch(cfgs, self.branch, len(cls_num))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.ModuleDict({task: nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                        nn.Linear(1280, cls_num[task]),) for task in self.tasks})
    def forward(self, x, task):
        feature = self.backbone(x, self.tasks.index(task))
        output = self.heads[task](torch.flatten(self.avgpool(feature),1)) 
        return output


############################################ EffNetV2 for Layout Models ######################################
class LayoutBlock(nn.Module):
    def __init__(self, block, task_set, layer_idx):
        super(LayoutBlock, self).__init__() 
        self.block = block
        self.task_set = task_set
        self.layer_idx = layer_idx
        
        # set outside 
        self.parent_block = [] 
        self.output = None
        
    def forward(self, x=None):
        if len(self.parent_block) == 0:
            if x is not None:
                return self.block(x)
            else:
                print('Wrong Input Data!')
                return
        else:
            return self.block(self.parent_block[0].output)
        
class EffNetV2_Backbone_Layout(nn.Module):
    def __init__(self, cfgs, layout, width_mult=1., pretrained=True):
        super(EffNetV2_Backbone_Layout, self).__init__()
        # Step 1: Sanity Check - The number of blocks in EffNetV2 should be 42
        if layout.B != sum([cfgs[i][2] for i in range(len(cfgs))])+2:
            print('Given layout cannot construct multi-task model of EffNetV2 because of the incompatiable number of blocks.')
            return
        
        self.cfgs = cfgs
        self.layout = layout
        
        # Step 2: Initiate Backbone Blocks - Modules in self.blocks are not used in forward
        self.blocks = []
        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        firstblock = conv_3x3_bn(3, input_channel, 2)
        self.__add_to_blocks(firstblock)
        # building inverted residual blocks
        block = MBConv
        total_stage_blocks = sum(cfg[2] for cfg in self.cfgs)
        stage_block_id = 0
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                sd_prob = 0.2 * float(stage_block_id) / total_stage_blocks
                blocklayer = block(input_channel, output_channel, s if i == 0 else 1, t, use_se, (t==1), sd_prob)
                self.__add_to_blocks(blocklayer)
                input_channel = output_channel
                stage_block_id += 1
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        lastblock = conv_1x1_bn(input_channel, output_channel)
        self.__add_to_blocks(lastblock)
        
        if pretrained:
            temp_basic_blocks = nn.ModuleList(self.blocks)
            state_dict = torch.load('/work/lijunzhang_umass_edu/data/multibranch/checkpoint/DomainNet/init/efficientnet_v2_s-dd5fe13b.pth')
            new_state_dict = OrderedDict()
            for pre_key, eff_key in zip(state_dict,temp_basic_blocks.state_dict()):
                new_state_dict[eff_key] = state_dict[pre_key]
            temp_basic_blocks.load_state_dict(new_state_dict)
            self.blocks = list(temp_basic_blocks)
        
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
        
        if not pretrained:
            self.__initialize_weights()
    
    def __add_to_blocks(self, block):
        self.blocks.append(block)
        return
    
    def __initialize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    
    def forward(self, x, task_idx): 
        for block in self.mtl_blocks:
            if task_idx in block.task_set:
                block.output = block(x)
                if block.layer_idx == self.layout.B - 1:
                    feature = block.output
                    break
        return feature
    
    def extract_features(self, x, task_idx): 
        features = []
        for block in self.mtl_blocks:
            if task_idx in block.task_set:
                block.output = block(x)
                features.append(block.output.detach().cpu().numpy())
        return features
    
class EffNetV2_FC(nn.Module):
    def __init__(self, tasks, layout=None, branch=None, cls_num={}, verbose=True, pretrained=True):
        super(EffNetV2_FC, self).__init__()
        self.tasks = tasks
        
        # Note: Both layout and branch are fined ones (coarse to fined are converted outside the model initialization)
        if layout is not None:
            # Constrcut MTL-Model from layout
            self.layout = layout
        elif branch is not None:
            # Construct first-order MTL-Model from the branching point and the number of blocks
            if len(self.tasks) != 2:
                print('The number of tasks to construct the first-order layouts is not 2.', flush=True)
                exit()
            self.branch = branch
            self.layout = self.first_order_layout()
        else:
            # Missing params
            print('Missing params for constrcuting multi-task model.', flush=True)
            exit()
        
        if verbose:
            print('Construct EffNetV2 from Layout:', flush=True)
            print(self.layout, flush=True)
        
        
        cfgs = [
        # t, c, n, s, SE for EffNet_s
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
        ]
        self.backbone = EffNetV2_Backbone_Layout(cfgs, self.layout, pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.ModuleDict({task: nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                        nn.Linear(1280, cls_num[task]),) for task in self.tasks})
        
    def forward(self, x, task):
        feature = self.backbone(x, self.tasks.index(task))
        output = self.heads[task](torch.flatten(self.avgpool(feature),1)) 
        return output
    
    def extract_features(self, x, task):
        return self.backbone.extract_features(x, self.tasks.index(task))
    
    #  Construct layout for two tasks and branch point 
    def first_order_layout(self):
        S = []
        for i in range(42):
            if i < self.branch:
                S.append([set([0,1])])
            else:
                S.append([set([0]),set([1])])
        layout = Layout(2, 42, S)
        return layout