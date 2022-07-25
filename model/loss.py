import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .metric import dice_coeff, multiclass_dice_coeff


def XE_loss(output, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target)


def dice_loss(output: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert output.size() == target.size()
    output = F.sigmoid(output)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(output.squeeze(), target.squeeze(), reduce_batch_first=False)
