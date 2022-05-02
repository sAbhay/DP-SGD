import torch
from torch import nn

hinge_loss = nn.HingeEmbeddingLoss()
def hinge_loss_01(pred, target):
    hinge_target = torch.clone(target)
    hinge_target[torch.where(hinge_target == 0)] = -1
    return hinge_loss(pred, hinge_target)