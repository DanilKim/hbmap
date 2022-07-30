"""*** cell classification layer ***"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNet(nn.Module):
    """ GAP -> FC """

    def __init__(self, in_channel, mid_channel, num_class):
        super(CNet, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_channel, mid_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_channel, num_class)
    
    def forward(self, x):
        ax = self.GAP(x)
        sx = ax.squeeze(-1).squeeze(-1)
        x1 = self.fc1(sx)
        xr = self.relu(x1)
        return self.fc2(xr)