import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .metric import dice_coeff, multiclass_dice_coeff


def nll_loss(output, target):
    return F.nll_loss(output, target)


def XE_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)


def dice_loss(output: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert output.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(output, target, reduce_batch_first=True)

