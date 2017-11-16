import gzip
import numpy as np
import torch
import cPickle as pickle

def load_ubuntu_corpus():
    path='askubuntu-master/text_tokenized.txt.gz'
    
    with gzip.open(path) as f:
        lines = f.readlines()

    for line in lines:
        question_id, question_title, question_body = line.split('\t')
        yield question_id, question_title, question_body


for x in load_ubuntu_corpus():
    print(x[0])
