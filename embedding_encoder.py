#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018-present, Papercup Technologies Limited
# All rights reserved.

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

from layers import ConcreteDropoutLayer, Conv2d

#test commit
class ConvEmbeddingEncoder(nn.Module):
    def __init__(self, opt):
        super(ConvEmbeddingEncoder, self).__init__()
        self.input_size = 63
        self.embedding_size = opt.embedding_size
        self.hidden_size = 32
        self.fully_connected = 64

        stride = (1, 2)
        padding = 1
        self.conv = nn.ModuleList([
            Conv2d(1, self.hidden_size, 3, padding=padding, weight_norm=True),
            Conv2d(self.hidden_size, self.hidden_size, 3, padding=padding, weight_norm=True, stride=stride),
            Conv2d(self.hidden_size, self.hidden_size * 2, 3, padding=padding, weight_norm=True, stride=2),
            Conv2d(self.hidden_size * 2, self.hidden_size, 3, padding=padding, weight_norm=True, stride=2),
            # Conv2d(self.hidden_size*2, self.hidden_size*2, 3, padding=padding, weight_norm=True),
        ])
        self.dropout = nn.ModuleList([
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            # nn.Dropout2d(0.1),
        ])

        self.maxpool = torch.nn.MaxPool2d(3, stride=2)
        #self.gru = torch.nn.GRU(544, self.fully_connected, 1, bidirectional=True)
        self.gru = torch.nn.GRU(512, self.fully_connected, 1, bidirectional=True)
        self.h_0 = Parameter(torch.zeros(2, 1, self.fully_connected))

        #self.fc_out = nn.utils.weight_norm(nn.Linear(self.fully_connected * 2, self.embedding_size))
        # hack for RNN=False
        self.fc_out = nn.utils.weight_norm(nn.Linear(512, self.embedding_size))

        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softsign = nn.Softsign()

    @staticmethod
    def _sequence_mask(sequence_length):
        max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1) \
            .expand_as(seq_range_expand)
        return (seq_range_expand == seq_length_expand - 1).t()

    def forward(self, x, start=True, lengths=None):

        #eps = x.new(x.size()).normal_(0, 0.01) # TODO: is this needed?
        x = x #+ eps

        # [T, B, C] -> [T, 1, B, C]
        out = x.unsqueeze(1)

        # [T, 1, B, C] -> [B, 1, C, T]
        out = out.permute(2, 1, 3, 0)
        out.contiguous()

        # Randomize input
        ##out = F.pad(out, (20, 0, 0, 0), mode='replicate')
        ##rnd = np.random.randint(20)
        #lengths = lengths + (20 - rnd)
        ##out = out[:, :, :, rnd:]

        # [B, C, H, T]
        for i, conv in enumerate(self.conv):
            #if self.training:
            out = self.dropout[i](out)
            out = conv(out)

        # [B, C, H, T] -> [B, C*H, T]
        out = out.view(out.size(0),
                       -1,
                       out.size(3))

        # [B, C, T] -> [T, B, C]
        out = out.permute(2, 0, 1).contiguous()

        RNN = False # TODO - turn back!
        # [T, B, C]
        if RNN:
            if start:
                h_0 = self.h_0.expand(self.h_0.size(0),
                                      out.size(1),
                                      self.fully_connected)
            else:
                h_0 = Variable(self.h_n.data.new(self.h_n.size()))
                h_0.data = self.h_n.data.clone()
            h_0 = h_0.contiguous()
            out, self.h_n = self.gru(out, h_0)

        #def transform(lengths):
        #    return ((lengths.float() + 2 - 2 - 1) / 2 + 1).floor()

        #lengths = transform(lengths)
        #lengths = transform(lengths)
        #lengths = transform(lengths)

        #mask = self._sequence_mask(lengths.long()).unsqueeze(2)
        #mask_ = mask.expand_as(out)

        # [T, B, C] -> [B, C]
        time_norm = 'mean'
        #time_norm = 'max' # !!! just testing !!!
        if time_norm == 'mean':
            # out = torch.sum(out*mask_.float(), dim=0) / mask_.float().sum(0)
            out = torch.mean(out, dim=0)
        elif time_norm == 'max':
            out = torch.max(out, dim=0)[0]
        elif time_norm == 'last':
            out = out[-1, :, :]

        ident_s = torch.tanh(self.fc_out(out))

        ident_s = torch.renorm(ident_s, 2, dim=0, maxnorm=1.0)

        return ident_s

        # NOTES... try just using a conv net on the full feature (current implementation does a convnet on each chunk
    # of 100 frames, then puts it through a RNN where the state is passed from one chunk to the next. May have to
    # be more careful about sequence lengths, mean vs. maxpool etc.
    # current problem is that the training loss is going down well, but this isn't reflected in the evaluation,
    # either for validation set or even randomized test set data. The issue is that the training 'loss' is
    # incremented in chunks of 100 where the embedding changes from step to step and keeps updating the loss
    # through the TBPTT iterations


class EmbeddingLookUp(nn.Module):
    def __init__(self, orig_embedding, opt):
        super(EmbeddingLookUp, self).__init__()
        self.nspk = opt.nspk
        self.embedding_size = opt.embedding_size
        self.lut_s = nn.Embedding(self.nspk,
                                  self.embedding_size)
        self.lut_s.weight.data.copy_(torch.from_numpy(orig_embedding))
        self.lut_s.require_grad = False

    def forward(self, spkr):
        spkr_n = (spkr + int(random.random() * 240)) % 240
        ident = self.lut_s(spkr)
        ident_n = self.lut_s(spkr_n)
        if ident.dim() == 3:
            ident = ident.squeeze(1)
        if ident_n.dim() == 3:
            ident_n = ident_n.squeeze(1)

        return ident, ident_n


class Discriminator(nn.Module):
    def __init__(self, opt, activation='relu'):
        super(Discriminator, self).__init__()

        self.lan = 2
        self.embedding_size = opt.embedding_size
        self.hidden_size = 128

        self.num_layers = 2

        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'sigmoid':
            self.activ = nn.Sigmoid()

        self.dropout = nn.Dropout(0.3)

        self.fc_in = nn.utils.weight_norm(nn.Linear(self.embedding_size, self.hidden_size))
        self.fc_out = nn.utils.weight_norm(nn.Linear(self.hidden_size, 1))
        self.fc_hidden = nn.ModuleList([nn.Sequential(
            nn.utils.weight_norm(nn.Linear(self.hidden_size, self.hidden_size)),
            self.activ,
            self.dropout,
        ) for _ in range(0, self.num_layers)])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.softsign = nn.Softsign()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc_in(input)
        out = self.activ(out)
        out = self.dropout(out)
        for i, layer in enumerate(self.fc_hidden):
            out = layer(out)
        out = self.fc_out(out)
        out = self.sigmoid(out)

        return out

    def load_max_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print('Model state size: ', own_state[name].size())
                print('Saved state size: ', param.size())


class SpkrClassifier(nn.Module):
    def __init__(self, opt, activation='relu'):
        super(SpkrClassifier, self).__init__()

        self.spkr_num = opt.nspk
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size
        self.num_layers = 3

        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'sigmoid':
            self.activ = nn.Sigmoid()

        # self.fc_out = nn.Linear(self.embedding_size, self.spkr_num)

        self.fc_in = nn.utils.weight_norm(nn.Linear(self.embedding_size, self.hidden_size))
        self.fc_out = nn.utils.weight_norm(nn.Linear(self.hidden_size, self.spkr_num))
        self.fc_hidden = nn.ModuleList([nn.Sequential(
            nn.utils.weight_norm(nn.Linear(self.hidden_size, self.hidden_size)),
            self.activ,
        ) for _ in range(0, self.num_layers)])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.softsign = nn.Softsign()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc_in(x)
        out = self.activ(out)
        for i, layer in enumerate(self.fc_hidden):
            out = layer(out)
        out = self.fc_out(out)
        return out

    def load_max_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print('Model state size: ', own_state[name].size())
                print('Saved state size: ', param.size())
