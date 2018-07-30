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
import torch.nn as nn

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
from utils import create_output_dir, wrap, check_grad
from training_monitor import TrainingMonitor
import eval_curves as ec

import model_discriminator as md

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
parser.add_argument('--debug', type=bool, default=False,
                    help='Reduced size to help with debugging')
parser.add_argument('--num-epochs-discriminator', type=int, default=50,
                    help='Number of epochs for initial training of discriminator network')
parser.add_argument('--lambda-reconstruction-loss', type=float, default=1.,
                    help='Weighting to reconstruction loss in the total loss calculation')
parser.add_argument('--lambda-discriminator-loss', type=float, default=1.,
                    help='Weighting to discriminator loss in the total loss calculation')
parser.add_argument('--lambda-discriminator-loss-ent', type=float, default=0.,
                    help='Weighting to discriminator loss in the total loss calculation')
parser.add_argument('--gender-method', type=str, default='add',
                    help='Method of including gender: add or concat')
parser.add_argument("--lambda-schedule", type=float, default=1000.,
                    help="Progressively increase discriminators' lambdas (0 to disable)")

# init
args = parser.parse_args()
args.expNameRaw = args.expName
args.expName = os.path.join('checkpoints', args.expName)
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logging = create_output_dir(args)
vis = visdom.Visdom(env=args.expName)

if args.gender_method == 'concat':
    args.speaker_hidden_size = args.hidden_size - 2
else:
    args.speaker_hidden_size = args.hidden_size

# data
logging.info("Building dataset.")
train_dataset = NpzFolder(args.data + '/numpy_features', args.nspk == 1)
if args.debug:
    train_dataset.npzs = train_dataset.npzs[:1000]

train_loader = NpzLoader(train_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         shuffle=True)

valid_dataset = NpzFolder(args.data + '/numpy_features_valid', args.nspk == 1)
if args.debug:
    valid_dataset.npzs = valid_dataset.npzs[:500]

valid_loader = NpzLoader(valid_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True)

logging.info("Dataset ready!")


def train(model, criterion, optimizer, epoch, train_losses, speaker_info, discriminator, discriminator_criterion,
          discriminator_criterion_ent,
          discriminator_optimizer):
    total = 0
    model.train()
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    min_acc = 1.0

    #counter = 0

    #if counter == 0:
    # at the start of each epoch, begin by resetting and training discriminator from scratch
    # bit hacky - aim is to prevent the weights from growing too large
    #discriminator.reset()
    embeddings = model.encoder.lut_s.weight.cpu().data.numpy()
    train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
    discriminator, train_accuracy, valid_accuracy, train_loss, valid_loss = md.train_discriminator(
        discriminator, train_data, valid_data, discriminator_criterion,
        discriminator_optimizer,
        num_epochs=args.num_epochs_discriminator)

    # extract the speaker embeddings from the model
    embeddings = model.encoder.lut_s.weight.cpu().data.numpy()

    # save the embeddings
    np.save('start_embeddings', (embeddings, speaker_info))

    #counter += 1

    # loop through mini-batches
    for full_txt, full_feat, spkr in train_enum:
        # evaluate discriminator accuracy (all speakers)
        # embeddings = model.encoder.lut_s.weight[:-1, :].cpu().data.numpy()
        embeddings = model.encoder.lut_s.weight.cpu().data.numpy()
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        discriminator_accuracy = md.eval_discriminator_accuracy(discriminator, train_data, discriminator_criterion)

        # extract speaker embeddings for the speakers in this mini-batch
        u_spkr = np.unique(spkr.numpy())

        # get discriminator data split for speakers in this batch
        embeddings = model.encoder.lut_s.weight[u_spkr, :].cpu().data.numpy()
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])

        # now split the examples in the mini-batch into shorter sequences (seq_len) and loop through those
        batch_iter = TBPTTIter(full_txt, full_feat, spkr, args.seq_len)
        batch_total = 0
        reconstruction_loss_total = 0

        for txt, feat, spkr, start in batch_iter:
            input = wrap(txt)
            target = wrap(feat)
            spkr = wrap(spkr)

            # Zero gradients
            if start:
                optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

            # extract embeddings for the speakers in this mini-batch
            embeddings = model.encoder.lut_s.weight[u_spkr, :].cpu().data.numpy()

            # get discriminator training data for these speakers
            train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])

            # train discriminator
            discriminator, train_accuracy, valid_accuracy, train_loss, valid_loss = md.train_discriminator(
                discriminator, train_data, valid_data, discriminator_criterion,
                discriminator_optimizer,
                num_epochs=args.num_epochs_discriminator, b_print=False)

            # forward pass of loop model
            is_male = np.array(speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M')
            gender = is_male.astype(np.float)
            output, _ = model([input, spkr, gender], target[0], start)
            reconstruction_loss = criterion(output, target[0], target[1])

            # forward pass of latent discriminator
            embeddings = model.encoder.lut_s.weight[u_spkr, :]
            train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])
            discriminator_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion)
            discriminator_loss_ent = md.eval_discriminator(discriminator, train_data, discriminator_criterion_ent)

            lambda_schedule = min(1.0, 1.0*(epoch-1) / args.lambda_schedule)
            #lambda_schedule = 1

            # total loss
            loss = args.lambda_reconstruction_loss * reconstruction_loss \
                   + args.lambda_discriminator_loss * lambda_schedule * discriminator_loss \
                   + args.lambda_discriminator_loss_ent * lambda_schedule * discriminator_loss_ent

            # Backward pass
            loss.backward()
            if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                logging.info('Not a finite gradient or too big, ignoring.')
                optimizer.zero_grad()
                continue
            optimizer.step()

            # Keep track of loss
            batch_total += loss.data[0]
            reconstruction_loss_total += reconstruction_loss.data[0]

        # total loss across all sequences in the mini-batch
        batch_total = batch_total / len(batch_iter)
        reconstruction_loss_total = reconstruction_loss_total / len(batch_iter)
        total += batch_total

        # compute discriminator loss/accuracy over all speakers
        # embeddings = model.encoder.lut_s.weight[:-1,:]
        embeddings = model.encoder.lut_s.weight
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        disc_accuracy = md.eval_discriminator_accuracy(discriminator, train_data,
                                                       discriminator_criterion)  # TODO wrong criterion??
        disc_ent = md.eval_discriminator(discriminator, train_data,
                                         discriminator_criterion_ent)  # TODO wrong criterion??

        train_enum.set_description(
            'Train (total_loss %.2f, recon_loss %.2f, disc_loss %.2f, disc_ent %.3f, disc_acc %.3f,) epoch %d' %
            (batch_total, reconstruction_loss_total, discriminator_loss, disc_ent, disc_accuracy, epoch))

        # save the embeddings
        embeddings = model.encoder.lut_s.weight.cpu().data.numpy()
        save_embeddings(embeddings, speaker_info, disc_accuracy, discriminator_loss, min_acc, epoch, model,
                        train_losses)

    # total loss over full epoch
    avg = total / len(train_loader)
    train_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(train_losses),
                 X=torch.arange(1, 1 + len(train_losses)),
                 opts=dict(title="Train"),
                 win='Train loss ' + args.expName)

    logging.info('====> Train set loss: {:.4f}'.format(avg))

    return avg


def evaluate(model, criterion, epoch, eval_losses, speaker_info, discriminator, discriminator_criterion,
             discriminator_criterion_ent):
    total = 0
    reconstruction_loss_total = 0
    valid_enum = tqdm(valid_loader, desc='Valid epoch %d' % epoch)

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        is_male = np.array(speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M')
        gender = is_male.astype(np.float)
        output, _ = model([input, spkr, gender], target[0])
        reconstruction_loss = criterion(output, target[0], target[1])

        # embeddings = model.encoder.lut_s.weight[:-1, :]
        embeddings = model.encoder.lut_s.weight # could pull this outside of loop - no need to run for every batch
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        discriminator_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion)
        disc_loss = discriminator_loss.data[0]

        discriminator_loss_ent = md.eval_discriminator(discriminator, train_data, discriminator_criterion_ent)

        lambda_schedule = min(1.0, 1.0 * (epoch - 1) / args.lambda_schedule)

        # total loss
        loss = args.lambda_reconstruction_loss * reconstruction_loss \
               + args.lambda_discriminator_loss * lambda_schedule * discriminator_loss \
               + args.lambda_discriminator_loss_ent * lambda_schedule * discriminator_loss_ent

        total += loss.data[0]
        reconstruction_loss_total += reconstruction_loss.data[0]

        disc_accuracy = md.eval_discriminator_accuracy(discriminator, train_data, discriminator_criterion)

        ent_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion_ent)
        ent_loss = ent_loss.data[0]

        valid_enum.set_description(
            'Valid (loss %.2f, discrim_loss %.2f, reconstruction_loss %.2f, discrim_acc %.3f, ent_loss %.3f) epoch %d' %
            (loss.data[0], reconstruction_loss, discriminator_loss, disc_accuracy, ent_loss, epoch))

    avg = total / len(valid_loader)
    avg_reconstruction_loss = reconstruction_loss_total / len(valid_loader)

    eval_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(eval_losses),
                 X=torch.arange(1, 1 + len(eval_losses)),
                 opts=dict(title="Eval"),
                 win='Eval loss ' + args.expName)

    logging.info('====> Test set loss: {:.4f}'.format(avg))
    return avg, avg_reconstruction_loss, disc_loss, disc_accuracy, ent_loss


def save_checkpoints(model, discriminator, epoch, best_eval, train_losses, eval_losses, eval_loss):
    # save checkpoint for this epoch
    # I'm saving every epoch so I can compute evaluation metrics across the training curve later on
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (args.expName, epoch))
    torch.save([args, train_losses, eval_losses, epoch],
               '%s/args.pth' % (args.expName))
    torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (args.expName, epoch))

    if eval_loss < best_eval:
        # if this is the best model yet, save it as 'bestmodel'
        torch.save(model.state_dict(), '%s/bestmodel.pth' % (args.expName))
        torch.save(discriminator.state_dict(), '%s/discriminator_bestmodel.pth' % (args.expName))
        best_eval = eval_loss

    # also keep a running copy of 'lastmodel'
    torch.save(model.state_dict(), '%s/lastmodel.pth' % (args.expName))
    torch.save([args, train_losses, eval_losses, epoch],
               '%s/args.pth' % (args.expName))
    torch.save(discriminator.state_dict(), '%s/discriminator_lastmodel.pth' % (args.expName))


def save_embeddings(embeddings, speaker_info, disc_accuracy, discriminator_loss, min_acc, epoch, model, train_losses):
    if np.abs(disc_accuracy - 0.5) < 0.05:
        np.save('train_embeddings_random', (embeddings, speaker_info, disc_accuracy, discriminator_loss))

    if disc_accuracy < 0.05:
        np.save('train_embeddings_negative', (embeddings, speaker_info, disc_accuracy, discriminator_loss))

    if disc_accuracy < min_acc:
        np.save('train_embeddings_min', (embeddings, speaker_info, disc_accuracy, discriminator_loss))
        min_acc = disc_accuracy

        torch.save(model.state_dict(), '%s/minmodel.pth' % (args.expName))
        torch.save([args, train_losses, disc_accuracy, epoch],
                   '%s/args.pth' % (args.expName))

    np.save('train_embeddings_last', (embeddings, speaker_info, disc_accuracy, discriminator_loss))


def main():
    start_epoch = 1
    speaker_info = md.get_speaker_info_for_discriminator()

    # create VoiceLoop network
    model = Loop(args)
    model.cuda()

    # if required, load model from checkpoint
    if args.checkpoint != '':
        checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[3]
        #model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
        cp = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

        if not 'encoder.lut_g.weight' in cp:
            # if picking up a checkpoint from the baseline sim, need to hack the checkpoint a bit to make it fit the new model structure
            # add lut_g
            cp['encoder.lut_g.weight'] = nn.Embedding(2, 256, max_norm=1.0).weight.data

            # if using concat, need to remove 2 dims from lut_s
            if args.gender_method == 'concat':
                cp['encoder.lut_s.weight'] = cp['encoder.lut_s.weight'][:, :-2]



        model.load_state_dict(cp)

    criterion = MaskedMSE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # create gender classifier network
    discriminator = md.LatentDiscriminator(args.speaker_hidden_size)
    discriminator.cuda()

    #if args.checkpoint != '':
        #disc_checkpoint = args.checkpoint.replace("epoch_", "discriminator_epoch_")
        #discriminator.load_state_dict(torch.load(disc_checkpoint, map_location=lambda storage, loc: storage))

    # train with cross-entropy for now... (as per FaderNetworks paper - check)
    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    # ...also use an entropy metric on the softmax output
    discriminator_criterion_ent = md.HLoss()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # Keep track of losses & best model
    train_losses = []
    eval_losses = []
    best_eval = float('inf')

    # create TrainingMonitor for logging out training curves
    training_monitor = TrainingMonitor(file=args.expNameRaw,
                                       exp_name=args.expNameRaw,
                                       b_append=True,
                                       path='training_logs',
                                       columns=(
                                       'epoch', 'update_time', 'train_loss', 'valid_loss', 'valid_reconstruction_loss', 'disc_loss', 'disc_accuracy',
                                       'ent_loss')
                                       )

    # Begin!
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # train model
        train(model,
              criterion,
              optimizer,
              epoch,
              train_losses,
              speaker_info,
              discriminator,
              discriminator_criterion,
              discriminator_criterion_ent,
              discriminator_optimizer
              )

        # evaluate on validation set
        eval_loss, reconstruction_loss, disc_loss, disc_accuracy, ent_loss = evaluate(model, criterion, epoch, eval_losses, speaker_info,
                                                                 discriminator, discriminator_criterion,
                                                                 discriminator_criterion_ent)

        # chk, _, _, _ = ec.evaluate(model=model,
        #                                  criterion=criterion,
        #                                  epoch=epoch,
        #                                  loader=valid_loader,
        #                                  metrics=('loss')
        #                                  )

        save_checkpoints(model, discriminator, epoch, best_eval, train_losses, eval_losses, eval_loss)

        # evaluate on a randomised subset of the training set
        if epoch % args.eval_epochs == 0:
            train_eval_loader = ec.get_training_data_for_eval(data=args.data,
                                                              len_valid=len(valid_loader.dataset))

            train_loss = ec.evaluate(model=model,
                                     criterion=criterion,
                                     epoch=epoch,
                                     loader=train_eval_loader,
                                     speaker_info=speaker_info,
                                     discriminator=discriminator,
                                     discriminator_criterion=discriminator_criterion,
                                     metrics=('loss'),
                                     lambda_reconstruction_loss=args.lambda_reconstruction_loss,
                                     lambda_discriminator_loss=args.lambda_discriminator_loss,
                                     lambda_discriminator_loss_ent=args.lambda_discriminator_loss_ent,
                                     lambda_schedule=args.lambda_schedule
                                     )[0]
        else:
            train_loss = None

        # store loss metrics
        training_monitor.insert(epoch=epoch, valid_loss=eval_loss, train_loss=train_loss,
                                valid_reconstruction_loss=reconstruction_loss,
                                disc_loss=disc_loss, disc_accuracy=disc_accuracy, ent_loss=ent_loss)
        training_monitor.write()


if __name__ == '__main__':
    main()
