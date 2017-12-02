import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from dataloader import create_variable
import numpy as np

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

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, avg_pool=True):
#         """
#         Inputs:
#         input_size = word embedding dimension
#         hidden_size = The number of features in the hidden state h
#         num_layers = number of hidden layers in LSTM
#         avg_pool - if avg_pool is True, then summarize hidden states by average pooling. If false, then take last hidden state
#         TO DO: implement things so that avg_pool works. Right now we are always taking the last hidden state
#         """
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.num_layers = num_layers

#     def init_hidden(self, batch_size):
#         h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)) 
#         c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
#         return (h0, c0)

#     def forward(self,x,sq_lengths):
#         sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
#         index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x)

#         sorted_inputs = x.gather(0, index_sorted_idx.long())
#         # pack sequence
#         packed_seq = nn.utils.rnn.pack_padded_sequence(
#                 # sorted_inputs, sorted_len.data.numpy(), batch_first=True)
#                 sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
#         # pass it to the lstm
#         out, self.hidden = self.lstm(packed_seq)

#         # unsort the output
#         _, original_idx = sorted_idx.sort(0, descending=False)

#         unsorted_idx = original_idx.view(1,-1,1).expand_as(self.hidden)
#         # put the hidden states back in the order that corresponds to the original sentence order
#         self.hidden = self.hidden.gather(1,  unsorted_idx.long())
#         return self.hidden[0] # doesn't work with num_layers > 1 currently. must figure this out.

#         # _, h_n, _ = self.lstm(emb) # h_n is the hidden state
#         # return h_n

#     def get_embed(self, seq):
#         self.hidden = self.init_hidden()
#         seq_w2v = Variable(seq)
#         return self(seq_w2v)

class LSTMRetrieval(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, batch_size=1):
        super(LSTMRetrieval, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = create_variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = create_variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)
        
    def forward(self, seq_tensor, seq_lengths):
        seq_tensor = create_variable(seq_tensor)
        if torch.cuda.is_available():
            seq_lengths = seq_lengths.cuda()
        packed_input, perm_idx = pack((seq_tensor, seq_lengths))
        # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # unpack your output if required
        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        _, orig_idx = perm_idx.sort(0, descending=False)
        return ht[-1][orig_idx] # Return last hidden layer, after unsorting the batch

    def get_embed(self, seq_tensor, seq_lengths):
        self.hidden = self.init_hidden()
        return self(seq_tensor, seq_lengths)
