# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import visdom
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
from utils import create_output_dir, wrap, check_grad
from training_monitor import TrainingMonitor
import evaluate_loss_func_for_notebook as el
import eval_mcd as eval_mcd
import pickle
import eval_curves as ec


parser = argparse.ArgumentParser(description='PyTorch Loop')
# Env options:
parser.add_argument('--epochs', type=int, default=92, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, default='vctk', metavar='E',
                    help='Experiment name')
parser.add_argument('--data', default='data/vctk',
                    metavar='D', type=str, help='Data path')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--gpu', default=0,
                    metavar='G', type=int, help='GPU device ID')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize train and validation loss.')
parser.add_argument('--eval-epochs', type=int, default=1,
                    help='how regularly to calculate evaluation metrics')
# Data options
parser.add_argument('--seq-len', type=int, default=100,
                    help='Sequence length for tbptt')
parser.add_argument('--max-seq-len', type=int, default=1000,
                    help='Max sequence length for tbptt')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--clip-grad', type=float, default=0.5,
                    help='maximum norm of gradient clipping')
parser.add_argument('--ignore-grad', type=float, default=10000.0,
                    help='ignore grad before clipping')
# Model options
parser.add_argument('--vocabulary-size', type=int, default=44,
                    help='Vocabulary size')
parser.add_argument('--output-size', type=int, default=63,
                    help='Size of decoder output vector')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='Hidden layer size')
parser.add_argument('--K', type=int, default=10,
                    help='No. of attention guassians')
parser.add_argument('--noise', type=int, default=4,
                    help='Noise level to use')
parser.add_argument('--attention-alignment', type=float, default=0.05,
                    help='# of features per letter/phoneme')
parser.add_argument('--nspk', type=int, default=22,
                    help='Number of speakers')
parser.add_argument('--mem-size', type=int, default=20,
                    help='Memory number of segments')

parser.add_argument('--checkpoint-utterance-embeddings', default='',
                    metavar='C', type=str, help='Checkpoint path for utterance embeddings')
parser.add_argument('--kld-lambda', type=float, default=1.,
                    help='Final weight to KLD loss component')
parser.add_argument('--kld-annealing-epochs', type=int, default=1,
                    help='KLD weight linearly rises to lambda over # epochs')
parser.add_argument('--kld-annealing-initial-epochs', type=int, default=0,
                    help='# epochs before KLD annealing begins (i.e. where lambda to KLD loss is zero')
parser.add_argument('--embedding-size', type=int, default=256,
                    help='Speaker embedding layer size')

# init
args = parser.parse_args()
args.expNameRaw = args.expName
args.expName = os.path.join('checkpoints', args.expName)
#args.embedding_size = args.hidden_size
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logging = create_output_dir(args)
vis = visdom.Visdom(env=args.expName)


# data
logging.info("Building dataset.")
train_dataset = NpzFolder(args.data + '/numpy_features', args.nspk == 1)
train_loader = NpzLoader(train_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         shuffle=True)

valid_dataset = NpzFolder(args.data + '/numpy_features_valid', args.nspk == 1)
valid_loader = NpzLoader(valid_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True)

logging.info("Dataset ready!")


def get_kld_lambda(epoch):
    if epoch > args.kld_annealing_initial_epochs:
        if args.kld_annealing_epochs == 1:
            kld_lambda = args.kld_lambda
        else:
            kld_lambda = min(1.0, 1.0 * (epoch - args.kld_annealing_initial_epochs - 1) / (args.kld_annealing_epochs - 1)) * args.kld_lambda
    else:
        # in initial epochs, ignore KLD term
        kld_lambda = 0.

    return kld_lambda


def train(model, criterion, optimizer, epoch, train_losses):
    total_mse = 0
    total_kld = 0
    total = 0   # Reset every plot_every
    model.train()
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    kld_lambda = get_kld_lambda(epoch)

    for full_txt, full_feat, spkr in train_enum:
        batch_iter = TBPTTIter(full_txt, full_feat, spkr, args.seq_len)
        batch_total_mse = 0
        batch_total_kld = 0
        batch_total = 0


        for txt, feat, spkr, start in batch_iter:
            input = wrap(txt)
            target = wrap(feat)
            spkr = wrap(spkr)

            # Zero gradients
            if start:
                optimizer.zero_grad()

            # Forward
            #output, _, _ = model([input, spkr], target[0], start)
            # do we want to build a single utterance embedding per full_feat, rather than one per 100-seq feat? (even if state
            # is preserved through the iterations?)
            output, _, _, ident_mu, ident_logvar = model([input, spkr], target[0], start=start, full_feat=wrap(full_feat[0]))
            loss_mse = criterion(output, target[0], target[1])

            #kld = -0.5 * torch.sum(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())
            # 09-Aug: change to mean. Def should be mean over the batch; less sure about mean over dimensions
            kld = -0.5 * torch.mean(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())

            loss = loss_mse + kld_lambda * kld

            # Backward
            loss.backward()
            if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                logging.info('Not a finite gradient or too big, ignoring.')
                optimizer.zero_grad()
                continue
            optimizer.step()

            # Keep track of loss
            batch_total_mse += loss_mse.data[0]
            batch_total_kld += kld.data[0]
            batch_total += loss.data[0]

        batch_total_mse = batch_total_mse / len(batch_iter)
        batch_total_kld = batch_total_kld / len(batch_iter)
        batch_total = batch_total/len(batch_iter)

        total_mse += batch_total_mse
        total_kld += batch_total_kld
        total += batch_total
        train_enum.set_description('Train (loss %.2f, MSE %.2f, KLD %.2f) epoch %d' %
                                   (batch_total, batch_total_mse, batch_total_kld, epoch))

    avg_mse = total_mse / len(train_loader)
    avg_kld = total_kld / len(train_loader)
    avg = total / len(train_loader)

    train_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(train_losses),
                 X=torch.arange(1, 1 + len(train_losses)),
                 opts=dict(title="Train"),
                 win='Train loss ' + args.expName)

    logging.info('====> Train set loss: {:.4f}'.format(avg))

    return avg, avg_mse, avg_kld


def evaluate(model, criterion, epoch, eval_losses):
    total_mse = 0
    total_kld = 0
    total = 0

    kld_lambda = get_kld_lambda(epoch)

    valid_enum = tqdm(valid_loader, desc='Valid epoch %d' % epoch)

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        output, _, _, ident_mu, ident_logvar = model([input, spkr], target[0])

        loss_mse = criterion(output, target[0], target[1])

        #kld = -0.5 * torch.sum(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())
        kld = -0.5 * torch.mean(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())

        loss = loss_mse + kld_lambda * kld

        total_mse += loss_mse.data[0]
        total_kld += kld.data[0]
        total += loss.data[0]

        valid_enum.set_description('Valid (loss %.2f, MSE %.2f, KLD %.2f) epoch %d' %
                                   (loss.data[0], loss_mse.data[0], kld.data[0], epoch))

    avg_mse = total_mse / len(valid_loader)
    avg_kld = total_kld / len(valid_loader)
    avg = total / len(valid_loader)
    eval_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(eval_losses),
                 X=torch.arange(1, 1 + len(eval_losses)),
                 opts=dict(title="Eval"),
                 win='Eval loss ' + args.expName)

    logging.info('====> Test set loss: {:.4f}'.format(avg))
    return avg, avg_mse, avg_kld


def main():
    start_epoch = 1
    model = Loop(args)
    model.cuda()

    if args.checkpoint != '':
        checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[3]
        cp = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

        if not 'embedding_encoder.conv.2.conv.weight_g' in cp:
            tmp = torch.load(args.checkpoint_utterance_embeddings, map_location=lambda storage, loc: storage)
            d = set(tmp.keys()) - set(cp.keys())
            for x in list(d):
                cp[x] = tmp[x]

        model.load_state_dict(cp)

    criterion = MaskedMSE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Keep track of losses
    train_losses = []
    eval_losses = []
    best_eval = float('inf')
    training_monitor = TrainingMonitor(file=args.expNameRaw,
                                       exp_name=args.expNameRaw,
                                       b_append=True,
                                       path='training_logs',
                                       columns=('epoch', 'update_time', 'train_loss', 'train_loss_mse', 'train_loss_kld', 'valid_loss', 'valid_loss_mse', 'valid_loss_kld', 'mcd')
                                       )

    # Begin!
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # train model
        train(model, criterion, optimizer, epoch, train_losses)

        # evaluate on validation set
        eval_loss, eval_mse, eval_kld = evaluate(model, criterion, epoch, eval_losses)

        #chk, _, _, _ = ec.evaluate(model=model,
        #                                  criterion=criterion,
        #                                  epoch=epoch,
        #                                  loader=valid_loader,
        #                                  metrics=('loss')
        #                                  )

        # save checkpoint for this epoch
        # I'm saving every epoch so I can compute evaluation metrics across the training curve later on
        torch.save(model.state_dict(), '%s/epoch_%d.pth' % (args.expName, epoch))
        torch.save([args, train_losses, eval_losses, epoch],
                   '%s/args.pth' % (args.expName))

        if eval_loss < best_eval:
            # if this is the best model yet, save it as 'bestmodel'
            torch.save(model.state_dict(), '%s/bestmodel.pth' % (args.expName))
            best_eval = eval_loss

        # also keep a running copy of 'lastmodel'
        torch.save(model.state_dict(), '%s/lastmodel.pth' % (args.expName))
        torch.save([args, train_losses, eval_losses, epoch],
                   '%s/args.pth' % (args.expName))

        # evaluate on a randomised subset of the training set
        if epoch % args.eval_epochs == 0:
            train_eval_loader = ec.get_training_data_for_eval(data=args.data,
                                                               len_valid=len(valid_loader.dataset))

            kld_lambda = get_kld_lambda(epoch)

            train_loss, train_loss_mse, train_loss_kld, _ ,_, _ = ec.evaluate(model=model,
                                                criterion=criterion,
                                                epoch=epoch,
                                                loader=train_eval_loader,
                                                metrics=('loss'),
                                                kld_lambda=kld_lambda
                                                )
        else:
            train_loss = None

        # store loss metrics
        training_monitor.insert(epoch=epoch, valid_loss=eval_loss, valid_loss_mse=eval_mse, valid_loss_kld=eval_kld,
                                train_loss=train_loss, train_loss_mse=train_loss_mse, train_loss_kld=train_loss_kld)
        training_monitor.write()


if __name__ == '__main__':
    main()
