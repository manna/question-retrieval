"""
A Script that combines Android-master/{dev|test}.pos.txt and
Android-master/{dev|test}.neg.txt into Android-master/{dev|test}.txt

The output files have the exact same format as "askubuntu-master/train_random.txt"
"""
from collections import OrderedDict

def load(mode, sign):
  with open('Android-master/{}.{}.txt'.format(mode, sign)) as f:
    return map(str.split, f.readlines())

def store_in_dict(lines):
  examples = OrderedDict()
  for query_id, other_id in lines:
    if query_id not in examples:
      examples[query_id] = []
    examples[query_id].append(other_id)
  return examples

def preprocess(mode):
  neg = store_in_dict(load(mode, 'neg'))
  pos = store_in_dict(load(mode, 'pos'))

  for query, pos_examples in pos.items():
    neg_examples = neg[query]
    line = '\t'.join([query, ' '.join(pos_examples), ' '.join(neg_examples)])
    yield line

for mode in ['dev', 'test']:
  with open('Android-master/{}.txt'.format(mode), 'w') as f:
    for line in preprocess(mode):
      f.write(line)