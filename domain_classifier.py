import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import create_variable

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, batch_size=None):
        """
        input_dim of the DomainClassifier will be the dim of the encodings
        produced by our Question encoding models.
        """
        super(DomainClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU()
        )

    def forward(self, x): return self.net(x)