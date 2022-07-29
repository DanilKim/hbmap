import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import F1Score


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_score(output: Tensor, target: Tensor, multiclass: bool = True):
    output = F.softmax(output, dim=1).float()
    target = F.one_hot(target, output.size(1)).permute(0, 3, 1, 2).float()

    assert output.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return fn(output, target, reduce_batch_first=False).item()


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1_score(output, target):
    with torch.no_grad():
        n_classes = output.size(1)
        device = output.get_device()
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1 = F1Score(num_classes=n_classes).to(device)
    return f1(pred, target)