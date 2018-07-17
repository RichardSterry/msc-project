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
from torch.autograd import Variable

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
from utils import create_output_dir, wrap, check_grad


def parse_args():
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

    parser.add_argument('--hidden-size-speakers', type=int, default=256,
                        help='Hidden layer size for speaker embeddings')

    args = parser.parse_args()
    args.expName = os.path.join('checkpoints', args.expName)

    return args


def init():
    # init
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args


def get_loader(data_path='data/vctk', max_seq_len=1000, batch_size=64, nspk=22):
    dataset = NpzFolder(data_path + '/numpy_features_valid', nspk == 1)
    loader = NpzLoader(dataset,
                             max_seq_len=max_seq_len,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)
    return loader


def stack_with_pad(in_list, constant_value=0, stack_axis=1, pad_axis=0):
    num_dim = in_list[0].ndim
    if pad_axis == -1:
        max_len = 0
    else:
        max_len = np.max([w.shape[pad_axis] for w in in_list])

    # pw = lambda x: [(0, max_len - x), (0,0), (0,0)]

    out_array = np.concatenate(
        [np.pad(w, pad_width=get_pad_width(w, num_dim, pad_axis, max_len), mode='constant',
                constant_values=constant_value)
         for w in in_list],
        axis=(stack_axis))

    return out_array


def get_pad_width(w, num_dim, pad_axis, max_len):
    pad_width = [(0,0) for i in range(num_dim)]
    if pad_axis > -1:
        pad_width[pad_axis] = (0, max_len - w.shape[pad_axis])
    return pad_width


def mse_manual(model_output, model_target, sequence_length):
    max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
                                       .expand_as(seq_range_expand)

    mask =  (seq_range_expand < seq_length_expand).t().float()

    mask = mask.unsqueeze(2)
    mask_ = mask.expand_as(model_target)
    loss = (model_output*mask_ - model_target*mask_)**2
    loss_avg = loss.sum() / mask.sum()
    loss_contrib = loss / mask.sum()
    #print mask.sum()
    assert (loss_contrib.sum() - loss_avg).abs().cpu().data.numpy() < 1e-4, "Loss contrib doesn't add up"
    return loss_avg, loss_contrib


def evaluate(model, loader, criterion, msg=''):
    total = 0
    my_total = 0
    valid_enum = tqdm(loader, desc='Calculating loss:' + msg)

    all_output = []
    all_txt = []
    all_txt_len = []
    all_target_feat = []
    all_target_len = []
    all_spkr = []
    all_loss_contrib = []
    all_attn = []

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        output, attn = model([input, spkr], target[0])
        loss = criterion(output, target[0], target[1])
        total += loss.data[0]

        my_loss, loss_contrib = mse_manual(output, target[0], target[1])
        my_total += my_loss.data[0]
        all_loss_contrib.append(loss_contrib.cpu().data.numpy() / len(loader)) # divide by number of batches


        all_output.append(output.cpu().data.numpy())
        all_txt.append(txt[0].cpu().numpy())
        all_txt_len.append(txt[1].cpu().numpy())
        all_target_feat.append(target[0].cpu().data.numpy())
        all_target_len.append(target[1].cpu().data.numpy())
        all_spkr.append(spkr.cpu().data.numpy().squeeze())

        all_attn.append(attn.cpu().data.numpy())
        valid_enum.set_description('Loss %.2f' % (loss.data[0]))

    avg = total / len(loader)
    my_avg = my_total / len(loader)

    loss_workings = dict()
    loss_workings['output'] = np.transpose(stack_with_pad(all_output, stack_axis=1, pad_axis=0), (1, 0, 2))
    loss_workings['txt'] = np.transpose(stack_with_pad(all_txt, stack_axis=1, pad_axis=0, constant_value=0), (1, 0))
    loss_workings['txt_len'] = stack_with_pad(all_txt_len, stack_axis=0, pad_axis=-1)
    loss_workings['target_feat'] = np.transpose(stack_with_pad(all_target_feat, stack_axis=1, pad_axis=0), (1, 0, 2))
    loss_workings['target_len'] = stack_with_pad(all_target_len, stack_axis=0, pad_axis=-1)
    loss_workings['spkr'] = stack_with_pad(all_spkr, stack_axis=0, pad_axis=-1)
    loss_workings['loss_contrib'] = np.transpose(stack_with_pad(all_loss_contrib, stack_axis=1, pad_axis=0), (1, 0, 2))

    # attn has variable length on two dimensions. Easier to store as a list.
    all_attn = [np.split(ary=w, indices_or_sections=w.shape[1], axis=1) for w in all_attn]
    loss_workings['attn'] = [val.squeeze() for sublist in all_attn for val in sublist]

    # add indices for batch number and position in batch
    y = [[(j, i) for i in range(x.shape[1])] for j, x in enumerate(all_output)]
    (loss_workings['idx_batch'], loss_workings['idx_pos_in_batch']) = zip(*[x for z in y for x in z])

    return avg, my_avg, loss_workings


def eval_loss(checkpoint='models/vctk/bestmodel.pth', data='data/vctk', max_seq_len=1000, nspk=22, gpu=0, batch_size=64, seed=1):
    #args = init()
    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print checkpoint
    print os.getcwd()
    checkpoint_args_path = os.path.dirname(checkpoint) + '/args.pth'
    checkpoint_args = torch.load(checkpoint_args_path)

    opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')
    train_args = opt[0]
    train_args.noise = 0
    train_args.checkpoint = checkpoint

    #args_to_use = args
    args_to_use = train_args

    print args_to_use
    model = Loop(args_to_use)

    model.cuda()
    model.load_state_dict(torch.load(args_to_use.checkpoint, map_location=lambda storage, loc: storage))

    criterion = MaskedMSE().cuda()

    loader = get_loader(data, max_seq_len, batch_size, nspk)

    eval_loss, my_eval_loss, loss_workings = evaluate(model, loader, criterion)

    print eval_loss
    print my_eval_loss

    return eval_loss, loss_workings


def main():
    args = init()

    checkpoint = args.checkpoint
    checkpoint_args_path = os.path.dirname(checkpoint) + '/args.pth'
    checkpoint_args = torch.load(checkpoint_args_path)

    opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')
    train_args = opt[0]
    train_args.noise = 0
    train_args.checkpoint = checkpoint

    args_to_use = args
    args_to_use = train_args

    print args_to_use
    model = Loop(args_to_use)

    model.cuda()
    model.load_state_dict(torch.load(args_to_use.checkpoint, map_location=lambda storage, loc: storage))

    criterion = MaskedMSE().cuda()

    loader = get_loader(args.data, args.max_seq_len, args.batch_size, args.nspk)

    eval_loss = evaluate(model, loader, criterion)

    print eval_loss

if __name__ == '__main__':
    main()
