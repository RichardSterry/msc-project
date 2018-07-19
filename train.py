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
import evaluate_loss_func_for_notebook as el
import eval_mcd as eval_mcd
import pickle
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



# init
args = parser.parse_args()
args.expNameRaw = args.expName
args.expName = os.path.join('checkpoints', args.expName)
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logging = create_output_dir(args)
vis = visdom.Visdom(env=args.expName)


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
          discriminator_optimizer,
          num_epochs_discriminator=500):

    total = 0   # Reset every plot_every
    model.train()
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    min_acc = 1.0

    counter = 0
    for full_txt, full_feat, spkr in train_enum:

        # train discriminator
        #discriminator.reset()
        #discriminator.cuda()

        u_spkr = np.unique(spkr.numpy())

        embeddings = model.encoder.lut_s.weight[:-1, :].cpu().data.numpy()
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        discriminator_accuracy = md.eval_discriminator_accuracy(discriminator, train_data, discriminator_criterion)

        #print "discriminator accuracy initialization: %0.3f" % discriminator_accuracy



        embeddings = model.encoder.lut_s.weight[u_spkr, :].cpu().data.numpy()

        #if embeddings.shape[0] == 108:
        #    embeddings = np.delete(embeddings, -1, axis=0)

        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])

        if counter == 0:
            discriminator,  train_accuracy, valid_accuracy, train_loss, valid_loss = md.train_discriminator(discriminator, train_data, valid_data, discriminator_criterion,
                                                   discriminator_optimizer,
                                                   num_epochs=50) #num_epochs_discriminator

            # extract the speaker embeddings from the model
            embeddings = model.encoder.lut_s.weight.cpu().data.numpy()

            # save the embeddings
            np.save('start_embeddings', (embeddings, speaker_info))

            counter += 1

        batch_iter = TBPTTIter(full_txt, full_feat, spkr, args.seq_len)
        batch_total = 0
        reconstruction_loss_total = 0

        for txt, feat, spkr, start in batch_iter:
            u_spkr2 = np.unique(spkr.numpy())
            assert np.all(u_spkr == u_spkr2)

            input = wrap(txt)
            target = wrap(feat)
            spkr = wrap(spkr)

            # Zero gradients
            if start:
                optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

            #embeddings = model.encoder.lut_s.weight.cpu().data.numpy()
            embeddings = model.encoder.lut_s.weight[u_spkr, :].cpu().data.numpy()

            # if embeddings.shape[0] == 108:
            #    embeddings = np.delete(embeddings, -1, axis=0)

            train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])
            discriminator, train_accuracy, valid_accuracy, train_loss, valid_loss = md.train_discriminator(
                discriminator, train_data, valid_data, discriminator_criterion,
                discriminator_optimizer,
                num_epochs=40, b_print=False)  # num_epochs_discriminator


            # Forward
            #gender = speaker_info[speaker_info.index == spkr.numpy()]
            is_male = np.array(speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M')
            gender = is_male.astype(np.float)
            output, _ = model([input, spkr, gender], target[0], start)
            reconstruction_loss = criterion(output, target[0], target[1])

            # latent discriminator
            #embeddings = model.encoder.lut_s.weight.cpu().data.numpy()
            #embeddings = model.encoder.lut_s.weight
            embeddings = model.encoder.lut_s.weight[u_spkr, :]

            #if embeddings.shape[0] == 108:
            #    embeddings = np.delete(embeddings, -1, axis=0)

            train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info[speaker_info.index.isin(u_spkr)])

            #discriminator_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion)
            discriminator_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion_ent)

            #loss = reconstruction_loss + 10 * discriminator_loss
            #loss = reconstruction_loss - 10 * discriminator_loss
            loss = -discriminator_loss
            #loss = reconstruction_loss + 5 * discriminator_loss
            #loss = discriminator_loss
            #loss = reconstruction_loss # should go back to default case...

            # Backward
            loss.backward()
            if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                logging.info('Not a finite gradient or too big, ignoring.')
                optimizer.zero_grad()
                continue
            optimizer.step()

            # Keep track of loss
            batch_total += loss.data[0]
            reconstruction_loss_total += reconstruction_loss.data[0]

        batch_total = batch_total/len(batch_iter)
        reconstruction_loss_total = reconstruction_loss_total/len(batch_iter)
        total += batch_total

        embeddings = model.encoder.lut_s.weight[:-1,:]
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        accuracy = md.eval_discriminator_accuracy(discriminator, train_data, discriminator_criterion) # TODO wrong criterion??
        disc_ent = md.eval_discriminator(discriminator, train_data,
                                                  discriminator_criterion_ent)  # TODO wrong criterion??

        train_enum.set_description('Train (total_loss %.2f, discrim_loss %.2f, reconstruction_loss %.2f, discrim_acc %.3f, disc_ent %.3f) epoch %d' %
                                   (batch_total, discriminator_loss, reconstruction_loss_total, accuracy, disc_ent, epoch))

        # extract the speaker embeddings from the model
        embeddings = model.encoder.lut_s.weight.cpu().data.numpy()

        # save the embeddings
        if np.abs(accuracy - 0.5) < 0.05:
            np.save('train_embeddings_random', (embeddings, speaker_info, accuracy, discriminator_loss))

        if accuracy < 0.05:
            np.save('train_embeddings_negative', (embeddings, speaker_info, accuracy, discriminator_loss))

        if accuracy < min_acc:
            np.save('train_embeddings_min', (embeddings, speaker_info, accuracy, discriminator_loss))
            min_acc = accuracy

            torch.save(model.state_dict(), '%s/minmodel.pth' % (args.expName))
            torch.save([args, train_losses, accuracy, epoch],
                       '%s/args.pth' % (args.expName))

        np.save('train_embeddings_last', (embeddings, speaker_info, accuracy, discriminator_loss))

    avg = total / len(train_loader)
    train_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(train_losses),
                 X=torch.arange(1, 1 + len(train_losses)),
                 opts=dict(title="Train"),
                 win='Train loss ' + args.expName)

    logging.info('====> Train set loss: {:.4f}'.format(avg))

    return avg


def evaluate(model, criterion, epoch, eval_losses, speaker_info, discriminator, discriminator_criterion, discriminator_criterion_ent):
    total = 0
    valid_enum = tqdm(valid_loader, desc='Valid epoch %d' % epoch)

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        is_male = np.array(speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M')
        gender = is_male.astype(np.float)
        output, _ = model([input, spkr, gender], target[0])
        reconstruction_loss = criterion(output, target[0], target[1])

        embeddings = model.encoder.lut_s.weight[:-1, :]
        train_data, valid_data = md.get_train_valid_split(embeddings, speaker_info)
        discriminator_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion)
        disc_loss = discriminator_loss.data[0]

        loss = reconstruction_loss# + 10 * discriminator_loss

        total += loss.data[0]

        disc_accuracy = md.eval_discriminator_accuracy(discriminator, train_data, discriminator_criterion)

        ent_loss = md.eval_discriminator(discriminator, train_data, discriminator_criterion_ent)
        ent_loss = ent_loss.data[0]

        valid_enum.set_description('Valid (loss %.2f, discrim_loss %.2f, reconstruction_loss %.2f, discrim_acc %.3f, ent_loss %.3f) epoch %d' %
                                   (loss.data[0], reconstruction_loss, discriminator_loss, disc_accuracy, ent_loss, epoch))


    avg = total / len(valid_loader)
    eval_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(eval_losses),
                 X=torch.arange(1, 1 + len(eval_losses)),
                 opts=dict(title="Eval"),
                 win='Eval loss ' + args.expName)

    logging.info('====> Test set loss: {:.4f}'.format(avg))
    return avg, disc_loss, disc_accuracy, ent_loss


def main():
    start_epoch = 1
    model = Loop(args)
    model.cuda()

    discriminator = md.LatentDiscriminator()
    discriminator.cuda()

    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    discriminator_criterion_ent = md.HLoss()

    if args.checkpoint != '':
        checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[3]
        model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))

    criterion = MaskedMSE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    speaker_info = md.get_speaker_info_for_discriminator()
    num_epochs_discriminator = 50


    # Keep track of losses
    train_losses = []
    eval_losses = []
    best_eval = float('inf')
    training_monitor = TrainingMonitor(file=args.expNameRaw,
                                       exp_name=args.expNameRaw,
                                       b_append=True,
                                       path='training_logs',
                                       columns=('epoch', 'update_time', 'train_loss', 'valid_loss', 'disc_loss', 'disc_accuracy', 'ent_loss')
                                       )

    # Begin!
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # train loop
        train(model, criterion, optimizer, epoch, train_losses, speaker_info, discriminator, discriminator_criterion,
              discriminator_criterion_ent,
             discriminator_optimizer, num_epochs_discriminator=num_epochs_discriminator)

        # evaluate on validation set
        eval_loss, disc_loss, disc_accuracy, ent_loss = evaluate(model, criterion, epoch, eval_losses, speaker_info, discriminator, discriminator_criterion, discriminator_criterion_ent)

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
                                                metrics=('loss')
                                                )[0]
        else:
            train_loss = None

        # store loss metrics
        training_monitor.insert(epoch=epoch, valid_loss=eval_loss, train_loss=train_loss,
                                disc_loss=disc_loss, disc_accuracy=disc_accuracy, ent_loss=ent_loss)
        training_monitor.write()


if __name__ == '__main__':
    main()
