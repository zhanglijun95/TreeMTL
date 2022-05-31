import torch
import torch.nn as nn

import google.protobuf.text_format
import sys
import copy 

from .layer_containers import LazyLayer

class ComputeNode(nn.Module):
    def __init__(self, protoLayer, fatherNodeList):
        super(ComputeNode, self).__init__()
        self.protoLayer = protoLayer
        # Get from bottom: fatherNode.top = CNode.bottom and fatherNode is the deepest one with the required top 
        self.fatherNodeList = fatherNodeList 
        self.nodeIdx = None # Set when appended into the ComputeNodes list in MTBlockModel
        
        self.layerParam = None # String: e.g., 'convolution_param' for nn.Conv2d
        self.paramMapping = {} # Dict: e.g., parameters in pytorch == parameters in prototxt for nn.Conv2d
        
        self.inputDim = None   # Int
        self.output = None
        self.outputDim = None   # Int
        
        self.basicOp = None
        self.layerName = None
        self.layerTop = None
        
    def build_layer(self):
        # Function: Build CNode from the protoLayer
        # Step 1: Match the protoLayer basic properties
        self.layerName = self.protoLayer.name
        self.layerTop = self.protoLayer.top
        self.set_input_channels()

        # Step 2: Match layer attributes
        #         Create params lists, e.g. ['in_channels=3', 'out_channels=32'] for nn.Conv2d
        torchParamList = []
        if bool(self.paramMapping):
            protoParamList = getattr(self.protoLayer, self.layerParam)
            for torchAttr in self.paramMapping:
                protoAttr = self.paramMapping[torchAttr]

                # Handle Input Dimension  
                if protoAttr == 'need_input':
                    torchParamList.append(torchAttr + '=' + str(self.inputDim))
                # Handle Operator Params
                else:   
                    protoAttrCont = getattr(protoParamList, protoAttr)
                    if isinstance(protoAttrCont, google.protobuf.pyext._message.RepeatedScalarContainer):
                        protoAttrCont = protoAttrCont[0]
                    torchParamList.append(torchAttr + '=' + str(protoAttrCont))

        # Step 3: Generate basic operators 
        self.generate_basicOp(torchParamList)
        self.set_output_channels()
        return
    
    def set_input_channels(self):
        # Function: Get input channels from the outputDim of fatherNodeList 
        #           inputDim: Int
        #           = fatherNodeList[0].outputDim [default for layers with 1 bottom]
        self.inputDim = self.fatherNodeList[0].outputDim
        return
    
    def set_output_channels(self):
        # Function: Set output channels according to different OpType
        #           outputDim: Int 
        #           = inputDim [default for layers that don't change dimensions]
        self.outputDim = self.inputDim
        return
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        return
    
    def reset_parameters(self):
        if hasattr(self.basicOp, 'reset_parameters'):
            self.basicOp.reset_parameters()
        return 
        
    def forward(self):
        return self.internal_compute()
        
    def internal_compute(self):
        # Function: Forward function when commonly used (shared with other tasks)
        #           [default for layers with 1 bottom]
        return self.basicOp(self.fatherNodeList[0].output)