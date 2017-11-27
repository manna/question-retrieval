import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size=1):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2)
        # self.pool = nn.AvgPool1d(kernel_size=2)

        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            self.pool
        )

    def forward(self, packed_seq, perm_idx):
        # TODO: cnn can't take PackedSequence

        out = self.cnn(seq)
        out = out.view(out.size(0), -1)
        
        return out

    get_embed = forward