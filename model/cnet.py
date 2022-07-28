"""*** cell classification layer ***"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNet(nn.Module):
    """ GAP -> FC """

    def __init__(self, num_class, in_channel):
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channel, num_class)
    
    def forward(self, x):
        x = self.GAP(x)
        return self.fc(x)