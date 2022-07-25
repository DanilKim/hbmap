import torch.nn.functional as F
from torch import Tensor
import torch

from model.metric import multiclass_dice_coeff, dice_coeff


def ce_loss(output: Tensor, target: Tensor):
    return F.cross_entropy(output, target)


def dice_loss(output: Tensor, target: Tensor, multiclass: bool = True):
    output = F.softmax(output, dim=1).float()
    target = F.one_hot(target, output.size(1)).permute(0, 3, 1, 2).float()
    
    # Dice loss (objective to minimize) between 0 and 1
    assert output.size() == target.size(), (output.size, target.size())
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(output, target, reduce_batch_first=True)


def ce_with_dice_loss(output: Tensor, target: Tensor):
    return ce_loss(output, target) + dice_loss(output, target, multiclass=True)