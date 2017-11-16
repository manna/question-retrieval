import gzip
import numpy as np
import torch
import cPickle as pickle

class Ubuntu():
    @staticmethod
    def load_corpus():
        """
        1.1 Setup.

        Load Q = {q1, ..., qn}
        """
        path='askubuntu-master/text_tokenized.txt.gz'
        
        with gzip.open(path) as f:
            lines = f.readlines()

        for line in lines:
            question_id, question_title, question_body = line.split('\t')
            yield question_id, question_title, question_body

    def load_training_data():
        """
        1.1 Setup

        Load a training set of similar question pairs {(q, q')}
        """
        pass
