import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import UbuntuDataset, batchify
from torch.autograd import Variable
from lstm_model import LSTMRetrieval

def train(train_loader, model, criterion, optimizer, epoch):
    print "Training..."
    print "Epoch {}".format(epoch)
    count = 0
    avg_loss = 0

    for i_batch, (padded_things, ys) in enumerate(train_loader):
        print("Batch #{}".format(i_batch)) 
        (qt_seq, qt_perm), (qb_seq, qb_perm), (ot_seq, ot_perm), (ob_seq, ob_perm) = padded_things

        # Step 1. Remember that Pytorch accumulates gradients. 
        # We need to clear them out before each instance
        model.zero_grad()
        
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        query_title = model.get_embed(qt_seq, qt_perm)
        query_body = model.get_embed(qb_seq, qb_perm)
        other_title = model.get_embed(ot_seq, ot_perm)
        other_body = model.get_embed(ob_seq, ob_perm)

        query_embed = (query_title + query_body) / 2
        other_embed = (other_title + other_body) / 2

        batch_avg_loss = criterion(query_embed, other_embed, ys)
        print "total (sum) loss for batch {} was {}".format(i_batch, batch_avg_loss.data)
        avg_loss += batch_avg_loss
        count += 1

        batch_avg_loss.backward()
        optimizer.step()

    avg_loss /= count
    print "average loss for epoch %i was %f"%(epoch,avg_loss)

def validation(val_loader, model, criterion):
    print "Validation..."
    count = 0
    avg_loss = 0

    for i_batch, (padded_things, ys) in enumerate(val_loader):
        print("Batch #{}".format(i_batch)) 
        (qt_seq, qt_perm), (qb_seq, qb_perm), (ot_seq, ot_perm), (ob_seq, ob_perm) = padded_things

        
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        query_title = model.get_embed(qt_seq, qt_perm)
        query_body = model.get_embed(qb_seq, qb_perm)
        other_title = model.get_embed(ot_seq, ot_perm)
        other_body = model.get_embed(ob_seq, ob_perm)

        query_embed = (query_title + query_body) / 2
        other_embed = (other_title + other_body) / 2

        batch_avg_loss = criterion(query_embed, other_embed, ys)
        print "total (sum) loss for batch {} was {}".format(i_batch, batch_avg_loss.data)
        avg_loss += batch_avg_loss
        count += 1

    avg_loss /= count
    print "average loss for validation was %f"%(avg_loss)



def main(args):
    if args.model_type == 'lstm':
        model = LSTMRetrieval(args.input_size, args.hidden_size, batch_size=args.batch_size)
    else: # otherwise model is a 'cnn'
        raise RuntimeError('Unknown --model_type')

    loss_function = nn.CosineEmbeddingLoss(margin=0, size_average=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # training_data = Ubuntu.load_training_data()
    print "Initializing Ubuntu Dataset..."
    train_dataset = UbuntuDataset(partition='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, # 100*n -> n questions.
        shuffle=False,
        num_workers=8,
        collate_fn=batchify
    )
    val_dataset = UbuntuDataset(partition='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # 100*n -> n questions.
        shuffle=False,
        num_workers=8,
        collate_fn=batchify
    )
    for epoch in xrange(args.epochs):
        train(train_dataloader, model, loss_function, optimizer, epoch)
        if epoch % args.val_epoch == 0:
            validation(val_dataloader, model, loss_function)




if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--model_type', default='lstm', type=str) # valid options are 'lstm' or 'cnn'
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--input_size', default=200, type=int)

    # training parameters
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.005, type=float)

    # miscellaneous
    parser.add_argument('--val_epoch', default=1, type=int)

    args = parser.parse_args()
    main(args)