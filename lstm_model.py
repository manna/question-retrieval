import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from dataloader import create_variable
import numpy as np
from IPython import embed

def pack( (seq_tensor, seq_lengths) ):
    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    # Otherwise, give (L,B,D) tensors
    seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)
    # print "seq_tensor ater transposing", seq_tensor.size() #, seq_tensor.data

    # pack them up nicely
    seq_lengths_numpy = np.maximum(seq_lengths.cpu().numpy(), 1) # eliminate any sequence lengths of 0
    packed_input = pack_padded_sequence(seq_tensor, seq_lengths_numpy)

    return (packed_input, perm_idx)

class LSTMRetrieval(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, num_layers, pool, batch_size=1):
        super(LSTMRetrieval, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.pool = pool
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_layers, batch_first=False)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = create_variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        c0 = create_variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        return (h0, c0)
        
    def forward(self, seq_tensor, seq_lengths):
        seq_tensor = create_variable(seq_tensor)
        if torch.cuda.is_available():
            seq_lengths = seq_lengths.cuda()
        packed_input, perm_idx = pack((seq_tensor, seq_lengths))
        # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # unpack your output if required
        output, sorted_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        _, orig_idx = perm_idx.sort(0, descending=False)
        if self.pool == "max":
            return ht[-1][orig_idx] # Return last hidden layer, after unsorting the batch
        else:
            output = torch.sum(output, dim=0)
            sorted_lengths_variable = torch.autograd.Variable(torch.Tensor(sorted_lengths)) # convert sorted_lengths list to variable
            if torch.cuda.is_available():
                sorted_lengths_variable = sorted_lengths_variable.cuda()
            avg_output = output * (1/sorted_lengths_variable).unsqueeze(1).expand(self.batch_size, self.hidden_dim)
            return avg_output
        # _, original_idx = perm_idx.sort(0, descending=False)

        # unsorted_idx = original_idx.view(1,-1,1).expand_as(self.hidden)
        # # put the hidden states back in the order that corresponds to the original sentence order
        # self.hidden = self.hidden.gather(1,  unsorted_idx.long())
        # return self.hidden[0] # doesn't work with num_layers > 1 currently. must figure this out.

        # unpacked, sorted_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)
        # unpacked = unpacked.transpose(0,1) # transpose added by me
        # unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
        # # we get the last index of each sequence
        # # idx = (seq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # idx = (seq_lengths-1).view(-1,1, 1).expand(unpacked.size(0), unpacked.size(1), unpacked.size(2))
        # embed()
        # # we sort and get the last element of each sequence
        # output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
        # output = output.view(output.size(0),output.size(1)*output.size(2))
        # return output

    def get_embed(self, seq_tensor, seq_lengths):
        self.hidden = self.init_hidden()
        return self(seq_tensor, seq_lengths)
