import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import UbuntuDataset, batchify, create_variable
from lstm_model import LSTMRetrieval
from cnn_model import CNN
from IPython import embed

def run_epoch(args, train_loader, model, criterion, optimizer, epoch, mode='train'):
    def get_top_results(num_results, top_results, new_result):
        """
        num_results = the number of results we want to keep among the top_results
        top_results = a list of tuples, in which first element of tuple is the positive or negative label of the question, the second is the score of that question
        new_result = a new result that we want to compare to the top_results. if it is good enough, put it in top_results
        """
        top_results.append(new_result)
        top_results = sorted(top_results, key=lambda i: i[1])[::-1] # sort results from greatest to least score
        top_results = top_results[:num_results] # keep only the top performing num_results number of results
        return top_results

    if mode == 'train':
        print "Training..."
    elif mode == 'val':
        print "Validation..."

    print "Epoch {}".format(epoch)
    count = 0
    total_loss = 0
    current_q_idx = 0 # represents the query index we are currently iterating over
    # current_q_best_score = -float('inf') # represents the score of the other question most similar to current query
    # current_q_best_score_correct = False # represents whether the best scoring other question is annotated as similar to the query
    current_q_top1 = []
    current_q_top5 = []
    top1_precision = 0.
    top5_precision = 0.
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
                current_q_top1_good_count = len([result for result in current_q_top1 if result[0] == 1])
                current_q_top5_good_count = len([result for result in current_q_top5 if result[0] == 1])
                top1_precision = (current_q_idx*top1_precision + current_q_top1_good_count)/float(current_q_idx + 1)
                top5_precision = (current_q_idx*top5_precision + current_q_top5_good_count)/float(5* (current_q_idx + 1))
                # top1_precision = (current_q_idx*top1_precision + current_q_best_score_correct)/float(current_q_idx + 1)
                # the numerator for top1_precision is the number of indices for which the best score 
                # the denominator for top1_precision is the number of unique query indices seen overall.
                current_q_idx = q_indices[i_element]
                current_q_top1 = []
                current_q_top5 = []

            element_score = criterion(query_embed[i_element:i_element+1], other_embed[i_element: i_element+1], torch.abs(ys[i_element: i_element+1]))
            current_q_top1 = get_top_results(1, current_q_top1, (ys.data[i_element], element_score.data[0]))
            current_q_top5 = get_top_results(5, current_q_top5, (ys.data[i_element], element_score.data[0]))
            # if element_score.data[0] > current_q_best_score:
            #     current_q_best_score = element_score.data[0]
            #     current_q_best_score_correct = (ys.data[i_element] == 1)

        print "total top1 precision seen so far until batch %i was %f"%(i_batch, top1_precision)
        print "total top5 precision seen so far until batch %i was %f"%(i_batch, top5_precision)

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
    loss_function = nn.CosineEmbeddingLoss(margin=0, size_average=False)
    
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

    args = parser.parse_args()
    main(args)
