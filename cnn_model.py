import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size=1):
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
        max_seq_length = seq_lengths.max()

        seq_tensor = seq_tensor.transpose(1,2)
        out = self.cnn(seq_tensor)
        assert out.size(2) == max_seq_length + 1 - self.window_size

        out = nn.AvgPool1d(kernel_size=out.size(2))(out)
        out = out.view(out.size(0), -1)

        return out

    get_embed = forward