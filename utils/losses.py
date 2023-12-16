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
        return loss


if __name__ == "__main__":
    alpha = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    pred = torch.randn(2, 5, 7, 12)
    target = torch.randint(size=(2, 5, 7, 12), low=0, high=5, dtype=torch.long).float()
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    print("CE default:", ce_loss_fn(pred, target)[:2, :2, :2])
    ce_loss_weighted_fn = torch.nn.CrossEntropyLoss(weight=alpha, reduction='none')
    print("CE weighted:", ce_loss_weighted_fn(pred, target)[:2, :2, :2])
    focal_loss_fn = FocalLoss()
    print("Focal default:", focal_loss_fn(pred, target)[:2, :2, :2])
    focal_loss_fn = FocalLoss(alpha=alpha)
    print("Focal with alpha:", focal_loss_fn(pred, target)[:2, :2, :2])
    focal_no_gamma_loss_fn = FocalLoss(gamma=0)
    print("Focal loss without gamma is the same as CE loss:", torch.allclose(focal_no_gamma_loss_fn(pred, target), ce_loss_fn(pred, target)))
