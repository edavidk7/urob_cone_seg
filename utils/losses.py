import torch
from torch.nn.functional import cross_entropy

def focal_loss(input, target, gamma=2):
    logp = -cross_entropy(input, target, reduction='none')
    loss = -(1 - torch.exp(logp)) ** self.gamma * logp
    return loss.mean()
