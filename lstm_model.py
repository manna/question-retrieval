import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from IPython import embed

from dataloader import UbuntuDataset, batchify

# class LSTM(nn.Module):
# 	def __init__(self, input_size, hidden_size, num_layers, avg_pool=True):
# 		"""
# 		Inputs:
# 		input_size = word embedding dimension
# 		hidden_size = The number of features in the hidden state h
# 		num_layers = number of hidden layers in LSTM
# 		avg_pool - if avg_pool is True, then summarize hidden states by average pooling. If false, then take last hidden state
# 		TO DO: implement things so that avg_pool works. Right now we are always taking the last hidden state
# 		"""
# 		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

# 	def init_hidden(self, sentence_length):
# 		return Variable(torch.zeros(1, 1, self.hidden_dim))

# 	def forward(self,x,sq_lengths):
# 		sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
# 		index_sorted_idx = sorted_idx\
# 				.view(-1,1,1).expand_as(x)
# 		sorted_inputs = x.gather(0, index_sorted_idx.long())
# 		# pack sequence
# 		packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
# 				# sorted_inputs, sorted_len.data.numpy(), batch_first=True)
# 				sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
# 		# pass it to the lstm
# 		out, hidden = self.lstm(packed_seq)

# 		# unsort the output
# 		_, original_idx = sorted_idx.sort(0, descending=False)

# 		unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
# 		unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
# 		# we get the last index of each sequence in the batch
# 		idx = (sq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
# 		# we sort and get the last element of each sequence
# 		output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
# 		output = output.view(output.size(0),output.size(1)*output.size(2))

# 		return output

# 		# _, h_n, _ = self.lstm(emb) # h_n is the hidden state
# 		# return h_n

class LSTMRetrieval(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMRetrieval, self).__init__()
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        c0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        return (h0, c0)
        
    def forward(self, embeds):
        lstm_out, self.hidden = self.lstm(embeds.view(len(embeds), 1, -1), self.hidden)
        # embed()
        return self.hidden[0] # I think this is right. We want to return h (the visible hidden state) not c (cell state). Might need to debug.

    def get_embed(self, seq):
        self.hidden = self.init_hidden()
        seq_w2v = Variable(seq)
        return self(seq_w2v)

loss_function = nn.CosineEmbeddingLoss()
model = LSTMRetrieval(200, 150)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# training_data = Ubuntu.load_training_data()
print "Initializing Ubuntu Dataset..."
ubuntu_dataset = UbuntuDataset()
dataloader = DataLoader(
    ubuntu_dataset,
    batch_size=1, # 100*n -> n questions.
    shuffle=False,
    num_workers=8,
    collate_fn=batchify
)

print "Training..."
for epoch in xrange(100): # again, normally you would NOT do 300 epochs, it is toy data
    print "Epoch {}".format(epoch)
    count = 0
    avg_loss = 0

    for i_batch, sample_batched in enumerate(dataloader):
        print "Batch #{}".format(i_batch)
        query_title, query_body, other_title, other_body, y = sample_batched
        
        # Step 1. Remember that Pytorch accumulates gradients. 
        # We need to clear them out before each instance
        model.zero_grad()
        
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        query_title = model.get_embed(query_title)
        query_body = model.get_embed(query_body)
        other_title = model.get_embed(other_title)
        other_body = model.get_embed(other_body)

        query_embed = (query_title + query_body) / 2
        other_embed = (other_title + other_body) / 2

        batch_avg_loss = loss_function(query_embed, other_embed, y)

        avg_loss += batch_avg_loss
        count += 1

        batch_avg_loss.backward()
        optimizer.step()

    avg_loss /= count
    print "average loss for epoch %i was %f"%(epoch,avg_loss)