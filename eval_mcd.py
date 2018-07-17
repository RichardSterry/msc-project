#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018-present, Papercup Technologies Limited
# All rights reserved.

import os
import argparse
import visdom

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop
from utils import create_output_dir, wrap, check_grad
from dtw import dtw




def model_def(checkpoint, gpu=-1, valid_loader=None):
    weights = torch.load(checkpoint,
                         map_location=lambda storage, loc: storage)
    opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')

    train_args = opt[0]
    train_args.noise = 0
    #norm = opt[5]
    #dict = {v: k for k, v in enumerate(code2phone)}
    norm = np.load(valid_loader.dataset.npzs[0])['audio_norminfo']

    model = Loop(train_args)
    model.load_state_dict(weights)

    if gpu >= 0:
        model.cuda()
    model.eval()

    return model, norm


logSpecDbConst = 10.0 / np.log(10.0) * np.sqrt(2.0)


def logSpecDbDist(x, y):
    diff = x - y
    return logSpecDbConst * np.sqrt(np.inner(diff, diff))


def evaluate(model, norm, valid_loader, logging=None):
    total = 0
    total1 = 0
    valid_enum = tqdm(valid_loader, desc='Valid')
    dtw_cost = logSpecDbDist

    cmp_mean = norm[0]
    cmp_std = norm[1]

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)
        #lan = wrap(lan)
        feat = target[0]

        # model.train()
        #feat = torch.FloatTensor(*target[0].size())
        #feat = Variable(target[0], volatile=True)
        #feat = wrap(feat)

        #with torch.no_grad():
        output, attn = model([input, spkr], feat)

        batch_size = attn.size(1)
        tmp_loss = 0

        ground_truths = target[0].cpu().data.numpy()
        ground_truths = ground_truths * cmp_std + cmp_mean

        synthesised = output.cpu().data.numpy()
        synthesised = synthesised * cmp_std + cmp_mean

        for i in range(batch_size):
            length = target[1][i].cpu().data.numpy()[0]
            ground_truth = ground_truths[:, i]
            ground_truth = ground_truth[:length, :25]

            synth = synthesised[:, i]
            synth = synth[:length, :25]

            unit_loss = dtw(ground_truth, synth, dtw_cost)
            unit_loss /= length
            tmp_loss += unit_loss
        tmp_loss /= batch_size

        total += tmp_loss
        valid_enum.set_description('Valid (MCD %.2f)' %
                                   (tmp_loss))

    avg = total / len(valid_loader)

    if logging:
        logging.info('====> Test set loss: {:.4f}'.format(avg))

    return avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch Loop')
    # Env options:
    parser.add_argument('--epochs', type=int, default=92, metavar='N',
                        help='number of epochs to train (default: 92)')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--expName', type=str, default='vctk', metavar='E',
                        help='Experiment name')
    parser.add_argument('--data', default='data/vctk',
                        metavar='D', type=str, help='Data path')
    parser.add_argument('--checkpoint', default='',
                        metavar='C', type=str, help='Checkpoint path')
    parser.add_argument('--gpu', default=0,
                        metavar='G', type=int, help='GPU device ID')
    # Data options
    parser.add_argument('--max-seq-len', type=int, default=1000,
                        help='Max sequence length for tbptt')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    # Model options
    parser.add_argument('--nspk', type=int, default=22,
                        help='Number of speakers')

    # init
    args = parser.parse_args()
    args.expName = os.path.join('checkpoints', args.expName)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging = create_output_dir(args)

    # data
    valid_dataset = NpzFolder(args.data + '/numpy_features_valid', args.nspk == 1)
    valid_loader = NpzLoader(valid_dataset,
                             max_seq_len=args.max_seq_len,
                             batch_size=args.batch_size,
                             num_workers=4,
                             pin_memory=True)

    # load model
    model, norm = model_def(args.checkpoint, gpu=args.gpu, valid_loader=valid_loader)

    # Begin!
    eval_loss = evaluate(model, norm, valid_loader, logging)


if __name__ == '__main__':
    main()