import gzip
import numpy as np
import torch
import cPickle as pickle
import collections

class Vectorizer():
    _VECTORS = None
    UNK = [0.]*200
    END_OF_TITLE = [1.]*200

    @staticmethod
    def load_pretrained_vectors():
        if Vectorizer._VECTORS is not None:
            return Vectorizer._VECTORS
        
        Vectorizer._VECTORS = {}
        path = 'askubuntu-master/vector/vectors_pruned.200.txt.gz'
        with gzip.open(path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            Vectorizer._VECTORS[line[0]] = map(float, line[1:])
        return Vectorizer

    @staticmethod
    def vectorize_word(w):
        """Returns the 200 dim word vector of w"""
        return Vectorizer._VECTORS.get(w, Vectorizer.UNK)

    @staticmethod
    def vectorize_sentence(words):
        """
        Returns a list of vectors of the words in the sentence 
        """
        return [Vectorizer.vectorize_word(w) for w in words] # + [Vectorizer.vectorize_word('<eos>')]

print "Loading pretrained word vectors..."
Vectorizer.load_pretrained_vectors()

# print(len(Vectorizer.vectorize_word('ubuntu')))
# print(len(Vectorizer.vectorize_sentence('ubuntu linux')))

class Ubuntu():
    _CORPUS = None

    @staticmethod
    def load_corpus():
        """
        1.1 Setup.

        Load Q = {q1, ..., qn}.
        Return python dictionary mapping id to Question.
        """
        if Ubuntu._CORPUS is not None:
            return Ubuntu._CORPUS

        Ubuntu._CORPUS = {}

        path = 'askubuntu-master/text_tokenized.txt.gz'
        with gzip.open(path) as f:
            lines = f.readlines()

        for line in lines:
            question_id, question_title, question_body = line.split('\t')
            
            title_words = question_title.split() # Todo is this the right tokenizer?
            body_words = question_body.split() 

            Ubuntu._CORPUS[question_id] = {
                'title': Vectorizer.vectorize_sentence(title_words),
                'body': Vectorizer.vectorize_sentence(body_words)
            }
        return Ubuntu._CORPUS

    @staticmethod
    def load_training_data():
        """
        1.1 Setup

        Load a training set of:

        query question, similar question ids, random question ids
        """
        path = 'askubuntu-master/train_random.txt'
        with open(path) as f:
            lines = f.readlines()
        
        CORPUS = Ubuntu.load_corpus()

        data = []
        for line in lines:
            # NOTE: similar_question_ids is a subset of random_question_ids
            query_question_id, similar_question_ids, random_question_ids = line.split('\t')
            data.append({
                'query_question' : CORPUS[query_question_id],
                'similar_questions' : [CORPUS[id] for id in similar_question_ids.split()],
                'random_questions' : [CORPUS[id] for id in random_question_ids.split()]
                })
        return data

from torch.utils.data import Dataset, DataLoader

to_title = lambda q: q['title']
to_body = lambda q: q['body']
to_title_and_body = lambda q: q['title'] + [Vectorizer.END_OF_TITLE] + q['body']

class UbuntuDataset(Dataset):
    def __init__(
        self,
        to_target_vec=to_title,
        to_context_vec=to_body
        ):
        """
        Loads the Ubuntu training dataset.

        self.data is a list of question groups.
        Each question group is a dictionary with keys:
            - query_question
            - similar_questions
            - random_questions

        similar_questions is a sublist of random_questions 
        """
        raw_data = Ubuntu.load_training_data()

        self.query_vecs = []
        self.other_vecs = []
        self.Y = [] # [1, 1, -1, -1, -1, 1, -1, -1 ....]
        self.len = 0 # = len(self.query_vecs) = len(self.other_vecs) = len(self.Y)

        for example in raw_data:
            query_vec = to_target_vec(example['query_question'])

            sim_count = len(example['similar_questions'])
            # The first sim_count random questions are similar.
            for i, other_q in enumerate(example['random_questions']):
                other_vec = to_context_vec(other_q)
                self.query_vecs.append(query_vec)
                self.other_vecs.append(other_vec)
                self.Y.append( 1 if i < sim_count else -1 )
                self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.query_vecs[idx], self.other_vecs[idx], self.Y[idx] 

print "Initializing Ubuntu Dataset..."
ubuntu_dataset = UbuntuDataset()

def pad(vectorized_seqs):
    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(map(len, vectorized_seqs)) # was torch.cuda.LongTensor

    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max(), 200))
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.Tensor(seq)

    return seq_tensor

    # If we wanted to return a PackedSequence object we could:

    # # SORT YOUR TENSORS BY LENGTH!
    # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # seq_tensor = seq_tensor[perm_idx]

    # # pack them up nicely
    # packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
    # return packed_input

    # # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
    # packed_output, (ht, ct) = lstm(packed_input)

    # # unpack your output if required
    # output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
    # print output

def pad_all(data):
    """
    @param data: [ ((query_vec, other_vec), label), ... ]
    """    
    query_vecs, other_vecs, Y = zip(*data)
    query_vecs = pad(query_vecs)
    other_vecs = pad(other_vecs)
    return zip(query_vecs, other_vecs, Y)

from torch.utils.data.dataloader import default_collate
def batchify(batch):
    padded_batch = pad_all(batch)
    return default_collate(padded_batch)

if __name__=='__main__':
    #Demo usage
    
    # Directly accessing ubuntu_dataset:
    # data_item = ubuntu_dataset[0]
    # print(data_item['query_question']) # Question(title, body)
    # print(len(data_item['similar_questions'])) # usually 1 or 2
    # print(len(data_item['random_questions'])) # 100

    # Accessing ubuntu_dataset using a DataLoader
    print "Loading Ubuntu Dataset..."
    dataloader = DataLoader(
        ubuntu_dataset, 
        batch_size=100, # 100*n -> n questions.
        shuffle=False,
        num_workers=8,
        collate_fn=batchify
    )

    for i_batch, sample_batched in enumerate(dataloader):
        print("batch #{}".format(i_batch)) 
        query_vec, other_vec, y = sample_batched
        print("query:")
        print(map(len, query_vec))
        print("other:")
        print(map(len, other_vec))
        print("y:")
        print(y)

        if i_batch == 50:
            break
        print("---------")
        