import torch
from torch import nn
from dataloader import Ubuntu
from torch.autograd import Variable
from IPython import embed

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

def get_embed(question, model):
    model.hidden = model.init_hidden()
    question_title_w2v = Variable(torch.Tensor(question.title))
    question_title_embed = model(question_title_w2v)
    model.hidden = model.init_hidden()
    question_body_w2v = Variable(torch.Tensor(question.body))
    question_body_embed = model(question_body_w2v)
    return (question_title_embed + question_body_embed)/2


loss_function = nn.CosineSimilarity(dim=2)
model = LSTMRetrieval(200, 150)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# training_data = Ubuntu.load_training_data()
for epoch in xrange(100): # again, normally you would NOT do 300 epochs, it is toy data
    count = 0
    avg_loss = 0
    for question, similar_questions, random_questions in Ubuntu.load_training_data():
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()
        
        # Also, we need to clear out the hidden state of the LSTM, detaching it from its
        # history on the last instance.
        question_embed = get_embed(question, model)
        similar_embed = get_embed(similar_questions[0], model)
        similar_cos = loss_function(question_embed, similar_embed)
        # print loss_function(question_embed, similar_embed)

        all_questions = similar_questions + random_questions
        max_similarity = Variable(torch.Tensor([-1]))
        for q in all_questions:
            q_embed = get_embed(q, model)
            cos = loss_function(question_embed, q_embed)
            if max_similarity.data[0] < cos.data[0]:
            	max_similarity = cos

        # print "the max similarity is"
        # print max_similarity
    
        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = max_similarity - similar_cos + (similar_cos == max_similarity).float() * 0.1 # this is the loss function
        # from the paper. Here the last term is a little hacky.
        avg_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        count += 1

    avg_loss /= count
    print "average loss for epoch %i was %f"%(epoch,avg_loss)