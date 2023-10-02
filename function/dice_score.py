import torch
import torch.nn as nn

class diceLoss(nn.Module):
    def __init__(self, n_channels, smooth = 1e-5):
        super(diceLoss, self).__init__()
        self.smooth = smooth
        self.n_channels = n_channels

    def partial_loss(self, pred, target):
        intersection = 2 * torch.sum(pred * target) + self.smooth
        union = torch.sum(pred * pred) + torch.sum(target * target) + self.smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, pred, target, weight = None, softmax = False):
        if softmax:
            pred = torch.softmax(pred, dim=1)
            target = torch.softmax(target, dim=1)
        if weight is None:
            weight = [1] * self.n_channels
        loss = 0
        for i in range(self.n_channels):
            loss += self.partial_loss(pred[:, i], target[:, i]) * weight[i]
        return loss / self.n_channels

