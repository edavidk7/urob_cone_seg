import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

def focal_loss(input, target, gamma=2):
    logp = -cross_entropy(input, target, reduction='none')
    loss = -(1 - torch.exp(logp)) ** gamma * logp
    return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2, alpha: Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        target = self.alpha * target
        logp = -cross_entropy(input, target, reduction='none')
        loss = -(1 - torch.exp(logp)) ** gamma * logp
        return focal_loss(input, target, self.gamma)

if __name__ == "__main__":
    # Test
    pred = torch.randn(2, 5, 7, 12)
    target = torch.randint(size=(2, 5, 7, 12), low=0, high=5, dtype=torch.long).float()
    print(pred.shape)
    print(target.shape)
    print(focal_loss(pred, target))
    # print(FocalLoss()(input, target))
    # print(FocalLoss(alpha=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))(input, target))
