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

from layers import Conv2d

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
        self.fc_out_mu = nn.utils.weight_norm(nn.Linear(512, self.embedding_size))

        self.fc_out_logvar = nn.utils.weight_norm(nn.Linear(512, self.embedding_size))

        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softsign = nn.Softsign()

    def forward(self, x, start=True, lengths=None):

        x = x

        # [T, B, C] -> [T, 1, B, C]
        out = x.unsqueeze(1)

        # [T, 1, B, C] -> [B, 1, C, T]
        out = out.permute(2, 1, 3, 0)
        out.contiguous()

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

        # [T, B, C] -> [B, C]
        time_norm = 'mean'
        #time_norm = 'max' # !!! just testing !!!
        if time_norm == 'mean':
            # out = torch.sum(out*mask_.float(), dim=0) / mask_.float().sum(0)
            out = torch.mean(out, dim=0)
        elif time_norm == 'max':
            out = torch.max(out, dim=0)[0]


        ident_mu = self.fc_out_mu(out)
        #ident_mu = torch.tanh(ident_mu)
        #ident_mu = torch.renorm(ident_mu, 2, dim=0, maxnorm=1.0)

        ident_logvar = self.fc_out_logvar(out)
        #ident_logvar = torch.tanh(ident_logvar)
        #ident_logvar = torch.renorm(ident_logvar, 2, dim=0, maxnorm=1.0)

        #ident_mu = self.fc_out_mu(out)
        #ident_logvar = self.fc_out_logvar(out)

        return ident_mu, ident_logvar

        # NOTES... try just using a conv net on the full feature (current implementation does a convnet on each chunk
    # of 100 frames, then puts it through a RNN where the state is passed from one chunk to the next. May have to
    # be more careful about sequence lengths, mean vs. maxpool etc.
    # current problem is that the training loss is going down well, but this isn't reflected in the evaluation,
    # either for validation set or even randomized test set data. The issue is that the training 'loss' is
    # incremented in chunks of 100 where the embedding changes from step to step and keeps updating the loss
    # through the TBPTT iterations

