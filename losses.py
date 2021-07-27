import torch.nn as nn
import torch
import torch.nn.functional as F

# Loss functions - adapted for use with Sen1Floods11 data from
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Differentiable equivalent of getting argmax, finds the predicted water labels
        inputs = F.softmax(inputs, dim=1)[:, 1]

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.view(-1)
        # Mask out "no data" sections from labels
        ignore = targets.ne(255)
        inputs = inputs.masked_select(ignore)
        targets = targets.masked_select(ignore)
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Differentiable equivalent of getting argmax, finds the predicted water labels
        inputs = F.softmax(inputs, dim=1)[:, 1]

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.view(-1)
        # Mask out "no data" sections from labels
        ignore = targets.ne(255)
        inputs = inputs.masked_select(ignore)
        targets = targets.masked_select(ignore)
        targets = targets.float()
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # Differentiable equivalent of getting argmax, finds the predicted water labels
        inputs = F.softmax(inputs, dim=1)[:, 1]

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.view(-1)
        # Mask out "no data" sections from labels
        ignore = targets.ne(255)
        inputs = inputs.masked_select(ignore)
        targets = targets.masked_select(ignore)
        targets = targets.float()
        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss