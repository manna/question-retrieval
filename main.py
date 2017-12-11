import argparse
import torch
# from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import UbuntuDataset, batchify, create_variable
from lstm_model import LSTMRetrieval
from cnn_model import CNN
import numpy as np
from collections import defaultdict

class MaxMarginCosineSimilarityLoss(_Loss):
    def __init__(self, examples_per_query=20):
        super(MaxMarginCosineSimilarityLoss, self).__init__()
        self.examples_per_query = examples_per_query

    def forward(self, queries, others, targets):
        """ 
        Requires that there is at least one positive example.

        If there is exactly 1 positive example:
            this loss function is exactly as described in arxiv 1512.05726
        """
        batch_size = targets.size()[0]
        min_positive_score = None
        max_score = None
        q_idx = 0
        max_margin_losses = [] # One for each unique query in the batch.

        for i in range(q_idx, batch_size):
            p = others[i]
            q = queries[i]
            score = F.cosine_similarity(q, p, dim=0)
            if (targets[i].data < 0).all(): # if -1
                score += 0.2 # delta
            if (targets[i].data > 0).all(): # if 1
                if min_positive_score is None:
                    min_positive_score = score
                else:
                    min_positive_score = min_positive_score.min(score)
            if max_score is None:
                max_score = score
            else:
                max_score = max_score.max(score)

            if (i+1) % self.examples_per_query == 0:
                max_margin_loss = max_score - min_positive_score
                max_margin_losses.append(max_margin_loss)

        return sum(max_margin_losses)

class QuestionRetrievalMetrics():
    def __init__(self):
        self.queries_count = 0

        self.top1_precision = 0.
        self.top5_precision = 0.
        self.MAP = 0.
        self.MRR = 0.

    def update(self, labels):
        top1_good_count = len([label for label in labels[:1] if label == 1])
        top5_good_count = len([label for label in labels[:5] if label == 1])
        
        if 1 in labels:
            # compute stats for MRR and MAP for current question
            first_positive_idx = labels.index(1)
            current_q_MRR = 1./(first_positive_idx+1)
            
            positive_indices = [question_idx for question_idx in range(len(labels)) if labels[question_idx]==1]
            current_q_MAP = np.mean([float(i+1)/(positive_indices[i]+1) for i in range(len(positive_indices))]) 
        else:
            current_q_MRR = 0.
            current_q_MAP = 0.

        self.top1_precision = get_moving_average(self.top1_precision, self.queries_count, 1, top1_good_count)
        self.top5_precision = get_moving_average(self.top5_precision, self.queries_count*5, 5, top5_good_count)
        self.MRR = get_moving_average(self.MRR, self.queries_count, 1, current_q_MRR)
        self.MAP = get_moving_average(self.MAP, self.queries_count, 1, current_q_MAP)

    def display(self, i_batch):
        print "average model top1 precision seen so far until batch %i was %f"%(i_batch, self.top1_precision)
        print "average model top5 precision seen so far until batch %i was %f"%(i_batch, self.top5_precision)
        print "average model MAP seen so far until batch %i was %f"%(i_batch, self.MAP)
        print "average model MRR seen so far until batch %i was %f"%(i_batch, self.MRR)

def get_moving_average(avg, num_prev_samples, num_new_samples, new_value):
    return float(avg*num_prev_samples + new_value)/(num_prev_samples + num_new_samples)

def update_metrics_for_batch(args, query_embed, other_embed, ys, mode, metrics, bm25_metrics):
    # Initialize some things at start of batch.
    model_q_results = [] # contains a list of tuples. first element of tuple is ground truth label (1 or -1) of question; 2nd is question score.
    if mode == "val":
        bm25_labels = []

    # Iterate through the batch to compute precision metrics
    for i in range(args.batch_size): 
        element_score = F.cosine_similarity(query_embed[i], other_embed[i], dim=0)
        model_q_results.append((ys.data[i], element_score.data[0]))
        if mode == "val":
            bm25_labels.append(ys.data[i])

        # Done with subbatch
        if (i + 1) % args.examples_per_query == 0:
            model_q_results = sorted(model_q_results, key=lambda result: result[1])[::-1]
            model_q_question_labels, _ = zip(*model_q_results)
            
            metrics.update(model_q_question_labels)
            model_q_results = [] # Clear model_q_results in prep for next subbatch
            metrics.queries_count += 1

            if mode == "val":
                bm25_metrics.update(bm25_labels)
                bm25_labels = []
                bm25_metrics.queries_count += 1 # queries seen in this epoch

def run_epoch(args, train_loader, model, criterion, optimizer, epoch, mode='train'):    
    queries_per_batch = args.batch_size/args.examples_per_query

    if mode == 'train':
        print "Training..."
    elif mode == 'val':
        print "Validation..."

    print "Epoch {}".format(epoch)
    total_loss = 0

    model_metrics = QuestionRetrievalMetrics()
    bm25_metrics = QuestionRetrievalMetrics() # Used in validation only

    for i_batch, (padded_things, ys) in enumerate(train_loader):
        print("Batch #{}".format(i_batch)) 
        ys = create_variable(ys)

        qt, qb, ot, ob = padded_things # padded_things might also be packed.
        # qt is (PackedSequence, perm_idx), or (seq_tensor, set_lengths)

        # Step 1. Remember that Pytorch accumulates gradients. 
        # We need to clear them out before each instance
        model.zero_grad()
        
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        query_title = model.get_embed(*qt)
        query_body = model.get_embed(*qb)
        other_title = model.get_embed(*ot)
        other_body = model.get_embed(*ob)

        query_embed = (query_title + query_body) / 2
        other_embed = (other_title + other_body) / 2


        if mode == "train":
            batch_loss = criterion(query_embed, other_embed, ys)
            total_loss += batch_loss.data[0]
            print "avg loss for batch {} was {}".format(i_batch, batch_loss.data[0]/queries_per_batch)
            batch_loss.backward()
            optimizer.step()

        update_metrics_for_batch(args, query_embed, other_embed, ys, mode, model_metrics, bm25_metrics)
        if i_batch % args.stats_display_interval == 0:
            model_metrics.display(i_batch)
            if mode == "val":
                print "BM25:"
                bm25_metrics.display(i_batch)


    avg_loss = total_loss / model_metrics.queries_count
    print "average {} loss for epoch {} was {}".format(mode, epoch, avg_loss)

def main(args):
    load, save, train, evaluate = args.load, args.save, not args.no_train, not args.no_evaluate
    del args.load
    del args.save
    del args.no_train
    del args.no_evaluate

    # MODEL

    if args.model_type == 'lstm':
        print "----LSTM----"
        model = LSTMRetrieval(args.input_size, args.hidden_size, args.num_layers, args.pool, batch_size=args.batch_size)
    elif args.model_type == 'cnn':
        print "----CNN----"
        model = CNN(args.input_size, args.hidden_size, args.pool, batch_size=args.batch_size)
    else:
        raise RuntimeError('Unknown --model_type')

    if load != '':
        print "Loading Model state from 'saved_models/{}'".format(load)
        model.load_state_dict(torch.load('saved_models/{}'.format(load)))

    # CUDA

    if torch.cuda.is_available():
        print "Using CUDA"
        model = model.cuda()
        model.share_memory()

    # Loss function and Optimizer
    loss_function = MaxMarginCosineSimilarityLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if load != '':
        print "Loading Optimizer state from 'saved_optimizers/{}'".format(load)
        optimizer.load_state_dict(torch.load('saved_optimizers/{}'.format(load)))
        optimizer.state = defaultdict(dict, optimizer.state) # https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536/5

    # Data

    # training_data = Ubuntu.load_training_data()
    print "Initializing Ubuntu Dataset..."
    if train:
        train_dataset = UbuntuDataset(name=args.dataset, partition='train')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, # 100*n -> n questions.
            shuffle=False,
            num_workers=8,
            collate_fn=batchify
        )
    if evaluate:
        val_dataset = UbuntuDataset(name=args.dataset, partition='dev')
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, # 100*n -> n questions.
            shuffle=False, 
            num_workers=8,
            collate_fn=batchify
        )

    for epoch in xrange(args.epochs):
        if train:
            run_epoch(args, train_dataloader, model, loss_function, optimizer, epoch, mode='train')
        if evaluate:
            if epoch % args.val_epoch == 0:
                run_epoch(args, val_dataloader, model, loss_function, optimizer, epoch, mode='val')

    if save:
        print "Saving Model state to 'saved_models/Model({}).pth'".format(args)
        torch.save(model.state_dict(), 'saved_models/Model({}).pth'.format(args))
        print "Saving Optimizer state to 'saved_optimizers/Optimizer({}).pth'".format(args)
        torch.save(optimizer.state_dict(), 'saved_optimizers/Optimizer({}).pth'.format(args))

if __name__=="__main__":
    """
    # Training
    $ python main.py --model_type lstm 
    $ python main.py --model_type cnn
    
    # Direct transfer, using pretrained model saved_models/M.pth
    $ python main.py --no-train --dataset android --load M.pth
    """

    parser = argparse.ArgumentParser()

    # loading and saving models. 'store_true' flags default to False. 
    parser.add_argument('--load', type=str, default='') # default is don't load.
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--no-evaluate', action='store_true')

    # model parameters
    parser.add_argument('--model_type', default='lstm', type=str, choices=['lstm', 'cnn'])
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--pool', default='max', type=str, choices=['max', 'avg'])

    # training parameters
    parser.add_argument('--dataset', default='ubuntu', type=str, choices=['ubuntu', 'android'])
    parser.add_argument('--batch_size', default=80, type=int) # constraint: batch_size must be a multiple of other_questions_size
    parser.add_argument('--examples_per_query', default=20, type=int) # the number of other questions that we want to have for each query
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.005, type=float)

    # miscellaneous
    parser.add_argument('--val_epoch', default=1, type=int)
    parser.add_argument('--stats_display_interval', default=1, type=int)

    args = parser.parse_args()
    main(args)
