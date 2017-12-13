from dataloader import Ubuntu
corpus_path='Android-master/corpus.tsv.gz'
CORPUS = Ubuntu.load_corpus(path=corpus_path)
examples_per_query = 20
example_divisor = 4

pos = dict()
for i, k in enumerate(CORPUS):
	if i % examples_per_query == 0:
		latest_k = k
		pos[latest_k] = [k]
	else:
	    pos[latest_k].append(k)

def preprocess(iters):
    total_length = iters * len(pos)
    nearest_length = total_length/example_divisor * example_divisor
    i = 0
    for _ in range(iters):
        for query, pos_examples in pos.items():
            neg_examples = ''
            line = '\t'.join([query, ' '.join(pos_examples), neg_examples])
            i += 1
            if i <= nearest_length:
                yield line

with open('Android-master/train.txt', 'w') as f:
    f.write('\n'.join(preprocess(5)))
