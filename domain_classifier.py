import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import create_variable

class GradientReversalLayer(nn.Module):
    def __init__(self, grl_constant):
        super(GradientReversalLayer, self).__init__()
        self.grl_constant = grl_constant

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -self.grl_constant * grad_output

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