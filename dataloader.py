import gzip
import torch
from IPython import embed

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
    def load_pretrained_vectors(path='askubuntu-master/vector/vectors_pruned.200.txt.gz'):
        if Vectorizer._VECTORS is not None:
            return Vectorizer._VECTORS
        
        Vectorizer._VECTORS = {}
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

class Ubuntu(): #TODO: Rename this.
    _CORPUS = {}

    @staticmethod
    def load_corpus(path='askubuntu-master/text_tokenized.txt.gz'):
        """
        1.1 Setup.

        Load Q = {q1, ..., qn}.
        Return python dictionary mapping id to Question.
        """
        if path in Ubuntu._CORPUS:
            return Ubuntu._CORPUS[path]

        Ubuntu._CORPUS[path] = {}
        
        with gzip.open(path) as f:
            lines = f.readlines()

        for line in lines:
            question_id, question_title, question_body = line.split('\t')
            
            title_words = question_title.split() # Todo is this the right tokenizer?
            body_words = question_body.split() 

            Ubuntu._CORPUS[path][question_id] = {
                'title': Vectorizer.vectorize_sentence(title_words),
                'body': Vectorizer.vectorize_sentence(body_words)
            }
        return Ubuntu._CORPUS[path]

    @staticmethod
    def load_eval_data(
        corpus_path='Android-master/corpus.tsv.gz',
        path_stem='Android-master/{}.{}.txt',
        dev_or_test='dev'
        ):

        CORPUS = Ubuntu.load_corpus(path=corpus_path)         
        data = []
        for partition, path in [
            ('similar_questions', path_stem.format(dev_or_test, 'pos')), 
            ('random_questions', path_stem.format(dev_or_test, 'neg'))
            ]:
            with open(path) as f:
                lines = f.readlines()

            for line in lines:
                query_question_id, other_question_id = line.split()
                obj = {
                    'query_question': CORPUS[query_question_id],
                    'similar_questions': [],
                    'random_questions': []
                    }
                obj[partition].append(CORPUS[other_question_id])
                data.append(obj)
        return data

    @staticmethod
    def load_training_data(
        corpus_path='askubuntu-master/text_tokenized.txt.gz',
        path='askubuntu-master/train_random.txt'
        ):
        """
        1.1 Setup

        Load a training set of:

        query question, similar question ids, random question ids
        """
        with open(path) as f:
            lines = f.readlines()
        
        CORPUS = Ubuntu.load_corpus(path=corpus_path)

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

    @staticmethod
    def load_ubuntu_eval_data(
        corpus_path='askubuntu-master/text_tokenized.txt.gz',
        path_stem='askubuntu-master/{}.txt',
        dev_or_test='dev' 
        ): # might want to make this same function as load_ubuntu_train_data
        """
        Load an evaluation set of:
        query question, similar question ids, random question ids, BM25 scores of random questions
        """
        with open(path_stem.format(dev_or_test)) as f:
            lines = f.readlines()
        
        CORPUS = Ubuntu.load_corpus(path=corpus_path)

        data = []
        for line in lines:
            query_question_id, similar_question_ids, random_question_ids, _  = line.split("\t")
            data.append({
                'query_question' : CORPUS[query_question_id],
                'similar_questions' : [CORPUS[id] for id in similar_question_ids.split()],
                'random_questions' : [CORPUS[id] for id in random_question_ids.split()]
                })

        return data

from torch.utils.data import Dataset, DataLoader

class UbuntuDataset(Dataset): 
    def __init__(self, examples_per_query=20, name='ubuntu', partition='train'):
        """
        Loads the Ubuntu training dataset.

        self.data is a list of question groups.
        Each question group is a dictionary with keys:
            - query_question
            - similar_questions
            - random_questions

        name, partition: ('ubuntu', 'train') | ('ubuntu', 'dev') | 
                         ('android', 'dev') | ('android', 'test')
        """
        self.query_titles = []
        self.query_bodies = []
        self.other_titles = []
        self.other_bodies = []
        self.Y = [] # [1, 1, -1, -1, -1, 1, -1, -1 ....]
        self.len = 0 # = len(self.query_vecs) = len(self.other_vecs) = len(self.Y)
        self.partition = partition
        if name == 'ubuntu':
            if self.partition == 'train':
                raw_data = Ubuntu.load_training_data()
            elif self.partition in {'dev', 'test'}:
                raw_data = Ubuntu.load_ubuntu_eval_data(dev_or_test=self.partition)
        elif name == 'android':
            raw_data = Ubuntu.load_training_data(
                corpus_path='Android-master/corpus.tsv.gz',
                path='Android-master/{}.txt'.format(self.partition)
            )
            # if self.partition == 'train':
            #     raise RuntimeError("No train data for android dataset")
            # elif self.partition in {'dev', 'test'}:
            #     raw_data = Ubuntu.load_training_data(
            #         corpus_path='Android-master/corpus.tsv.gz',
            #         path='Android-master/{}.txt'.format(self.partition)
            #     )


        if name == 'ubuntu' and self.partition in {"dev", "test"}:
            for example in raw_data:
                query_title = example['query_question']['title']
                query_body = example['query_question']['body']

                sim_count = len(example['similar_questions'])
                for i in range(examples_per_query):
                    other_q = example['random_questions'][i]
                    if other_q in example['similar_questions']:
                        self.Y.append(1)
                    else:
                        self.Y.append(-1)
                    other_title = other_q['title']
                    other_body = other_q['body']
                    self.query_titles.append(query_title)
                    self.query_bodies.append(query_body)
                    self.other_titles.append(other_title)
                    self.other_bodies.append(other_body)

                self.len += examples_per_query

        else:
            for example in raw_data:
                query_title = example['query_question']['title']
                query_body = example['query_question']['body']

                sim_count = len(example['similar_questions'])
                for i in range(examples_per_query):
                    if i < sim_count:
                        self.Y.append( 1 )
                        other_q = example['similar_questions'][i]
                    else:
                        try:
                            self.Y.append( -1 )
                            other_q = example['random_questions'][i]
                        except: # invalid assumption that there are args.examples_per_query random questions
                            break # (this happens in eval data)
                    other_title = other_q['title']
                    other_body = other_q['body']
                    self.query_titles.append(query_title)
                    self.query_bodies.append(query_body)
                    self.other_titles.append(other_title)
                    self.other_bodies.append(other_body)
                self.len += examples_per_query

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            return (self.query_titles[idx], self.query_bodies[idx],
                self.other_titles[idx], self.other_bodies[idx], self.Y[idx])
        except IndexError as e:
            print e
            return None

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
    data = filter(lambda x: x is not None, data) # Dataset.__getitem__ can return None
    q_titles, q_bodies, o_titles, o_bodies, ys = zip(*data)
    padded_things = map(pad, (q_titles, q_bodies, o_titles, o_bodies))
    return padded_things, torch.LongTensor(ys)

if __name__=='__main__':
    #Demo usage
    
    # Directly accessing ubuntu_dataset:
    # data_item = ubuntu_dataset[0]
    # print(data_item['query_question']) # Question(title, body)
    # print(len(data_item['similar_questions'])) # usually 1 or 2
    # print(len(data_item['random_questions'])) # 100

    # Accessing ubuntu_dataset using a DataLoader

    print "Loading Ubuntu Dataset..."
    ubuntu_dataset = UbuntuDataset(partition='test', name='android')
    dataloader = DataLoader(
        ubuntu_dataset, 
        batch_size=100, # 20*n -> n questions.
        shuffle=False,
        num_workers=0,
        collate_fn=batchify
    )

    for i_batch, (padded_things, ys) in enumerate(dataloader):
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
        
