import argparse
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import UbuntuDataset, batchify, create_variable
from lstm_model import LSTMRetrieval
from cnn_model import CNN
from IPython import embed

class MaxMarginCosineSimilarityLoss(_Loss):
    def __init__(self, margin=0):
        super(MaxMarginCosineSimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, queries, others, targets):
        """ 
        Assumes one query per batch: (all queries are the same)
        
        Weakly assumes only one positive example.
        In that case, this loss function is exactly as described in arxiv 1512.05726
        """
        batch_size = targets.size()[0]
        min_positive_score = None
        max_score = None
        for i in range(batch_size):
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
        max_margin_loss = max_score - min_positive_score
        return max_margin_loss

def run_epoch(args, train_loader, model, criterion, optimizer, epoch, mode='train'):
    def get_moving_average(avg, num_prev_samples, num_new_samples, new_value):
        return float(avg*num_prev_samples + new_value)/(num_prev_samples + num_new_samples)

    if mode == 'train':
        print "Training..."
    elif mode == 'val':
        print "Validation..."

    print "Epoch {}".format(epoch)
    count = 0
    total_loss = 0
    current_q_idx = 0 # represents the query index we are currently iterating over
    current_q_results = [] # contains a list of tuples. first element of tuple is ground truth label (1 or -1) of question; 2nd is question score.
    top1_precision = 0.
    top5_precision = 0.
    MAP = 0.
    MRR = 0.
    for i_batch, (q_indices, padded_things, ys) in enumerate(train_loader):
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

        batch_avg_loss = criterion(query_embed, other_embed, ys)
        total_loss += batch_avg_loss.data[0]
        count += 1
        print "total (sum) loss for batch {} was {}".format(i_batch, batch_avg_loss.data[0])
        
        for i_element in range(args.batch_size): 
            # for computing accuracy metrics
            if q_indices[i_element] != current_q_idx:
                current_q_results = sorted(current_q_results, key=lambda result: result[1])[::-1]
                current_q_top1_good_count = len([result for result in current_q_results[:1] if result[0] == 1])
                current_q_top5_good_count = len([result for result in current_q_results[:5] if result[0] == 1])
                current_q_question_labels, _ = zip(*current_q_results)

                # compute stats for MRR and MAP for current question
                first_positive_idx = current_q_question_labels.index(1)
                current_q_MRR = 1./(first_positive_idx+1)
                
                positive_indices = [question_idx for question_idx in range(len(current_q_question_labels)) if current_q_question_labels[question_idx]==1]
                current_q_MAP = np.mean([float(i+1)/(positive_indices[i]+1) for i in range(len(positive_indices))]) 

                top1_precision = get_moving_average(top1_precision, current_q_idx, 1, current_q_top1_good_count)
                top5_precision = get_moving_average(top5_precision, 5*current_q_idx, 5, current_q_top5_good_count)
                MRR = get_moving_average(MRR, current_q_idx, 1, current_q_MRR)
                MAP = get_moving_average(MAP, current_q_idx, 1, current_q_MAP)

                current_q_idx = q_indices[i_element]
                current_q_results = []

            element_score = criterion(query_embed[i_element:i_element+1], other_embed[i_element: i_element+1], torch.abs(ys[i_element: i_element+1]))
            current_q_results.append((ys.data[i_element], element_score.data[0]))

        if i_batch % args.stats_display_interval == 0:
            print "current q_idx: %i"%current_q_idx
            print "average top1 precision seen so far until batch %i was %f"%(i_batch, top1_precision)
            print "average top5 precision seen so far until batch %i was %f"%(i_batch, top5_precision)
            print "average MAP seen so far until batch %i was %f"%(i_batch, MAP)
            print "average MRR seen so far until batch %i was %f"%(i_batch, MRR)

        if mode == 'train':
            batch_avg_loss.backward()
            optimizer.step()

    avg_loss = total_loss / count
    print "average {} loss for epoch {} was {}".format(mode, epoch, avg_loss)

def main(args):
    if args.model_type == 'lstm':
        print "----LSTM----"
        model = LSTMRetrieval(args.input_size, args.hidden_size, batch_size=args.batch_size)
    elif args.model_type == 'cnn':
        print "----CNN----"
        model = CNN(args.input_size, args.hidden_size, batch_size=args.batch_size)
    else:
        raise RuntimeError('Unknown --model_type')
    
    if torch.cuda.is_available():
        print "Using CUDA"
        model = model.cuda()
        model.share_memory()
    loss_function = MaxMarginCosineSimilarityLoss(margin=0.2)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # training_data = Ubuntu.load_training_data()
    print "Initializing Ubuntu Dataset..."
    train_dataset = UbuntuDataset(partition='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, # 100*n -> n questions.
        shuffle=False, # if shuffle=True, accuracy metrics will get screwed up
        num_workers=8,
        collate_fn=batchify
    )
    val_dataset = UbuntuDataset(partition='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # 100*n -> n questions.
        shuffle=False, # if shuffle=True, accuracy metrics will get screwed up
        num_workers=8,
        collate_fn=batchify
    )
    for epoch in xrange(args.epochs):
        run_epoch(args, train_dataloader, model, loss_function, optimizer, epoch, mode='train')
        if epoch % args.val_epoch == 0:
            run_epoch(args, val_dataloader, model, loss_function, optimizer, epoch, mode='val')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--model_type', default='lstm', type=str, choices=['lstm', 'cnn'])
    parser.add_argument('--hidden_size', default=150, type=int)
    parser.add_argument('--input_size', default=200, type=int)

    # training parameters
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.005, type=float)

    # miscellaneous
    parser.add_argument('--val_epoch', default=1, type=int)
    parser.add_argument('--stats_display_interval', default=1, type=int)

    args = parser.parse_args()
    main(args)
