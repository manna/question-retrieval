import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size=None):
        super(CNN, self).__init__()

        self.window_size = 4
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=self.window_size),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )

    def forward(self, seq_tensor, seq_lengths):
        # seq_tensor.size() is (batch_size, seq_length, input_size=200)
        # We need it to be (batch_size, input_size=200, seq_length)
        seq_tensor = seq_tensor.transpose(1,2)

        if seq_lengths.max() < self.window_size:
            # In pytorch 0.4, this is as simple as `seq_tensor = F.pad(seq_tensor, (left, right), "constant", 0)`
            seq_tensor4d = seq_tensor.unsqueeze(2) # add fake dimension
            seq_tensor4d = F.pad(seq_tensor4d, (0, self.window_size-1, 0, 0), "constant", 0)
            seq_tensor = seq_tensor4d.squeeze(2) # remove fake height
            
        out = self.cnn(seq_tensor)
        assert (out.size(2) == seq_lengths.max() # if we padded
                or out.size(2) == seq_lengths.max() - self.window_size + 1) # if we didn't pad

        out = nn.AvgPool1d(kernel_size=out.size(2))(out)

        out = out.view(out.size(0), -1)
        return out

    get_embed = forward