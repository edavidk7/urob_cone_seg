import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.nn import CrossEntropyLoss
from . import N_CLASSES


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 1, weight: Tensor = None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def __call__(self, input, target):
        logp = -cross_entropy(input, target, reduction='none')
        if self.alpha is not None:
            logp *= torch.max(self.alpha.view(1, -1, 1, 1) * target, dim=1)[0]
        loss = -(1 - torch.exp(logp))**self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassDistrToWeight:
    @staticmethod
    def sqrt_one_minus(w: Tensor) -> Tensor:
        return torch.sqrt(1 - w)

    @staticmethod
    def one_minus(w: Tensor) -> Tensor:
        return 1 - w

    @staticmethod
    def reciprocal(w: Tensor) -> Tensor:
        return 1 / w

    @staticmethod
    def reciprocal_with_class(w: Tensor) -> Tensor:
        return 1 / (w * N_CLASSES)


if __name__ == "__main__":
    alpha = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    # pred -> B x C x H x W logit tensor
    pred = torch.randn(2, 5, 7, 12) * 10
    # target -> B x C x H x W one hot tensor
    target = torch.zeros(2, 5, 7, 12)
    for b in range(2):
        for h in range(7):
            for w in range(12):
                target[b, torch.randint(0, 5, (1,)), h, w] = 1.0

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    print("CE default:", ce_loss_fn(pred, target))
    ce_loss_weighted_fn = torch.nn.CrossEntropyLoss(weight=alpha, reduction='mean')
    print("CE weighted:", ce_loss_weighted_fn(pred, target))
    focal_loss_fn = FocalLoss()
    print("Focal default:", focal_loss_fn(pred, target).mean())
    focal_loss_fn = FocalLoss(weight=alpha)
    print("Focal weighted:", focal_loss_fn(pred, target).mean())
    focal_no_gamma_loss_fn = FocalLoss(gamma=0)
    print("Focal loss without gamma is the same as CE loss:", torch.allclose(focal_no_gamma_loss_fn(pred, target).mean(), ce_loss_fn(pred, target)))
