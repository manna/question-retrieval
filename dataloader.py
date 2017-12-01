import gzip
import numpy as np
import torch
import cPickle as pickle
import collections
import sys

from torch.autograd import Variable
def create_variable(tensor):
    """
    TODO: Move to utils.py
    """
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    return Variable(tensor)

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

class UbuntuDataset(Dataset):
    def __init__(self, partition='train'):
        """
        Loads the Ubuntu training dataset.

        self.data is a list of question groups.
        Each question group is a dictionary with keys:
            - query_question
            - similar_questions
            - random_questions

        similar_questions is a sublist of random_questions 

        partition: valid options are 'train' or 'val'.
        """
        raw_data = Ubuntu.load_training_data()
        self.query_indices = []
        self.query_titles = []
        self.query_bodies = []
        self.other_titles = []
        self.other_bodies = []
        self.Y = [] # [1, 1, -1, -1, -1, 1, -1, -1 ....]
        self.len = 0 # = len(self.query_vecs) = len(self.other_vecs) = len(self.Y)
        self.partition = partition
        if self.partition == "train":
            start_index = 0
            end_index = len(raw_data)-20000
        elif self.partition == 'val':
            start_index = len(raw_data)-20000
            end_index = len(raw_data)

        for query_idx, example in enumerate(raw_data[start_index:end_index]):
            query_title = example['query_question']['title']
            query_body = example['query_question']['body']

            sim_count = len(example['similar_questions'])
            for i in range(100):
                if i < sim_count:
                    self.Y.append( 1 )
                    other_q = example['similar_questions'][i]
                else:
                    self.Y.append( -1 )
                    other_q = example['random_questions'][i]
                other_title = other_q['title']
                other_body = other_q['body']
                self.query_indices.append(query_idx)
                self.query_titles.append(query_title)
                self.query_bodies.append(query_body)
                self.other_titles.append(other_title)
                self.other_bodies.append(other_body)
            self.len += 100

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.query_indices[idx], self.query_titles[idx], self.query_bodies[idx],
                self.other_titles[idx], self.other_bodies[idx], self.Y[idx])

def pad(vectorized_seqs, embedding_size=200):
    vectorized_seqs = list(vectorized_seqs)
    seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs])
    seq_tensor = torch.zeros(
        (len(vectorized_seqs), seq_lengths.max(), embedding_size)
    )

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        if seqlen > 0:
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)

    return (seq_tensor, seq_lengths)


def batchify(data):
    q_indices, q_titles, q_bodies, o_titles, o_bodies, ys = zip(*data)

    padded_things = map(pad, (q_titles, q_bodies, o_titles, o_bodies))
    
    return torch.LongTensor(q_indices), padded_things, torch.LongTensor(ys),  


if __name__=='__main__':
    #Demo usage
    
    # Directly accessing ubuntu_dataset:
    # data_item = ubuntu_dataset[0]
    # print(data_item['query_question']) # Question(title, body)
    # print(len(data_item['similar_questions'])) # usually 1 or 2
    # print(len(data_item['random_questions'])) # 100

    # Accessing ubuntu_dataset using a DataLoader
    print "Loading Ubuntu Dataset..."
    ubuntu_dataset = UbuntuDataset()
    dataloader = DataLoader(
        ubuntu_dataset, 
        batch_size=3, # 100*n -> n questions.
        shuffle=False,
        num_workers=0,
        collate_fn=batchify
    )
    sys.exit(0)

    for i_batch, (q_indices, padded_things, ys) in enumerate(dataloader):
        print("batch #{}".format(i_batch)) 
        (qt_seq, qt_perm), (qb_seq, qb_perm), (ot_seq, ot_perm), (ob_seq, ob_perm) = padded_things

        print("query titles:")
        print(qt_seq)
        print("qt perm:")
        print(qt_perm)
        print("other:")
        print(ot_seq)
        print("y:")
        print(ys)

        if i_batch == 50:
            break
        print("---------")
        
