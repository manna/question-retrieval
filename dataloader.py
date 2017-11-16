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

Vectorizer.load_pretrained_vectors()
# print(len(Vectorizer.vectorize_word('ubuntu')))
# print(len(Vectorizer.vectorize_sentence('ubuntu linux')))

class Question():
    def __init__(self, title, body):
        self.title = title
        self.body = body

    def __repr__(self):
        return "Question" + repr((self.title, self.body))

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
            Ubuntu._CORPUS[question_id] = Question(
                Vectorizer.vectorize_sentence(question_title), 
                Vectorizer.vectorize_sentence(question_body)
                ) 
        
        return Ubuntu._CORPUS

    @staticmethod
    def load_training_data():
        """
        1.1 Setup

        Load a training set of:

        query question, similar question pairs {(q, q')}
        """
        path = 'askubuntu-master/train_random.txt'
        with open(path) as f:
            lines = f.readlines()
        
        CORPUS = Ubuntu.load_corpus()

        for line in lines:
            # NOTE: similar_question_ids is a subset of random_question_ids
            query_question_id, similar_question_ids, random_question_ids = line.split('\t')
            yield ( CORPUS[query_question_id] 
                  , [CORPUS[id] for id in similar_question_ids.split()]
                  , [CORPUS[id] for id in random_question_ids.split()]  )

# for q, s, r in Ubuntu.load_training_data():
#     pass
