import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from dataloader import create_variable

class GradReverse(Function):
    grl_constant = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * GradReverse.grl_constant

 
class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.f = GradReverse()

    def forward(self, x):
        return self.f.apply(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, batch_size=None):
        """
        input_dim of the DomainClassifier will be the dim of the encodings
        produced by our Question encoding models.
        """
        super(DomainClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, 1),
        #     nn.ReLU()
        # )


    def forward(self, x): return self.net(x).squeeze(1)