"""*** cell classification layer ***"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNet(nn.Module):
    """ GAP -> FC """

    def __init__(self, num_class, in_channel):
        super(CNet, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_class, in_channel)
    
    def forward(self, x):
        ax = self.GAP(x)
        sx = ax.squeeze(-1).squeeze(-1)
        return self.fc(sx)