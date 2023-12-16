import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2, alpha: Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, input, target):
        logp = -cross_entropy(input, target, reduction='none')
        loss = -(1 - torch.exp(logp)) ** self.gamma * logp
        if self.alpha is not None:
            target = self.alpha.view(1, -1, 1, 1) * target
            target_scales = torch.max(target, dim=1)[0]
            loss = target_scales * loss
        return loss.mean()

if __name__ == "__main__":
    pred = torch.randn(2, 5, 7, 12)
    target = torch.randint(size=(2, 5, 7, 12), low=0, high=5, dtype=torch.long).float()
    print(pred.shape, target.shape)
    focal_loss_fn = FocalLoss()
    print(focal_loss_fn(pred, target))
    focal_loss_fn = FocalLoss(alpha=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
    print(focal_loss_fn(pred, target))
