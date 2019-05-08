from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from torch.nn.functional import binary_cross_entropy_with_logits

# Loss function for training filter
def filter_loss(outputs, targets):
    targets_encoded = torch.FloatTensor(outputs.shape).fill_(1.).cuda()
    
    _, inds = outputs.max(1)
    
    for i in range(targets.shape[0]):
        if inds[i] == targets[i]:
            # If properly classified, set as a target all classes but correct class
            targets_encoded[i, targets[i]] = 0.
        else:
            # If misclassified, set as a target misclassified class
            targets_encoded[i] = 0.
            targets_encoded[i, inds[i]] = 1.

    loss = torch.mean(binary_cross_entropy_with_logits(outputs, torch.ones(outputs.shape).cuda()))
    return loss

# First attempt at filter for ResNet. Had some issues: use ConvFilter2
class ConvFilter(torch.nn.Module):
    
    def __init__(self, filter_weight=.3):
        super(ConvFilter, self).__init__()
        
        self.filter_weight = filter_weight
        
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(64)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.tanh5 = nn.Tanh()
        
        # in channels, out channels, kernel_size, padding=n
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 16, 1, padding=0)
        self.conv4 = nn.Conv2d(16, 64, 1, padding=0)
        self.conv5 = nn.Conv2d(64, 3, 3, padding=1)
        
        
        # self.limit = 256 // 20;

    def generate_filters(self, x):
    
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.norm1(y)
        
        y = self.conv2(y)
        y_ = self.relu2(y)
        y_ = self.norm2(y_)
        
        y_ = self.conv3(y_)
        y_ = self.relu3(y_)
        y_ = self.norm3(y_)
        
        y_ = self.conv4(y_) + y
        y = self.relu4(y_)
        y = self.norm4(y)
        
        y = self.conv5(y)
        y = self.tanh5(y)
        
        return y

    def forward(self, x):
        
        y = self.generate_filters(x)
        
        y = torch.max(y, torch.FloatTensor(y.shape).fill_(-self.filter_weight).cuda())
        y = torch.min(y, torch.FloatTensor(y.shape).fill_(self.filter_weight).cuda())
        
        y = x + y
        
        # y = torch.max(y, self.zeros[:,0:y.shape[-2],0:y.shape[-1]])
        # y = torch.min(y, self.ones[:,0:y.shape[-2],0:y.shape[-1]])
        
        return y

# Final implementation of a filter for ResNet
class ConvFilter2(torch.nn.Module):
    
    def __init__(self, filter_weight=.1):
        super(ConvFilter2, self).__init__()
        
        # How much is filter allowed to modify image.
        # Given how data is normalized, .4 is approx 25 px, or 10 percent intensity
        self.filter_weight = filter_weight
        
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(32)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.tanh3 = nn.Tanh()
        
        self.conv1 = nn.Conv2d(13, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)


    # function to generate unweighted filters
    def generate_filters(self, x):
    
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.norm1(y)
        
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.norm2(y)
        
        y = self.conv3(y)
        y = self.tanh3(y)
        
        return y

    # Forward function. Generates filter, applies it to image after weighting
    # x[:,:3] sould be RGB images
    # x[:,i+3] = 1. for true target class, 0. otherwise
    def forward(self, x):
        
        y = self.generate_filters(x) * self.filter_weight
        
        y = x[:,0:3] + y
        
        return y
    
    
