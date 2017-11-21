import gzip
import numpy as np
import torch
import cPickle as pickle

class Vectorizer():
    _VECTORS = None
    UNK = [0.]*200

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
    def vectorize_sentence(s):
        """
        Returns a list of vectors of the words in the sentence 
        """
        return [Vectorizer.vectorize_word(w) for w in s.split()] # + [Vectorizer.vectorize_word('<eos>')]

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
            Ubuntu._CORPUS[question_id] = {
                'title': Vectorizer.vectorize_sentence(question_title),
                'body': Vectorizer.vectorize_sentence(question_body)
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
    def __init__(self):
        """
        Loads the Ubuntu training dataset.

        self.data is a list of question groups.
        Each question group is a dictionary with keys:
            - query_question
            - similar_questions
            - random_questions

        similar_questions is a sublist of random_questions 
        """
        self.data = Ubuntu.load_training_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print "Initializing Ubuntu Dataset..."
ubuntu_dataset = UbuntuDataset()

"""
for i in range(4): # or range(len(ubuntu_dataset))
    data_item = ubuntu_dataset[i]
    print(data_item['query_question']) # Question(title, body)
    print(len(data_item['similar_questions'])) # usually 1 or 2
    print(len(data_item['random_questions'])) # 100
"""

# print "Loading Ubuntu Dataset..."
# dataloader = DataLoader(ubuntu_dataset, batch_size=64,
#                         shuffle=True, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['query_question'].size(),
#           sample_batched['similar_questions'].size())
#     break