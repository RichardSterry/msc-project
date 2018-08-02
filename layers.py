import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class glu(nn.Module):
    def __init__(self, dim=-1):
        super(glu, self).__init__()
        self.dim = dim

    def forward(self, input):
        output = F.glu(input, dim=self.dim)

        return output

def getLinear(dim_in, dim_out, weight_norm=True):
    in_layer = nn.Linear(dim_in, dim_in // 10)
    out_layer = nn.Linear(dim_in // 10, dim_out)
    if weight_norm:
        in_layer = nn.utils.weight_norm(in_layer)
        out_layer = nn.utils.weight_norm(out_layer)
    return nn.Sequential(
        in_layer,
        nn.ReLU(),
        out_layer,
    )

class ConcreteDropoutLayer(nn.Module):
    def __init__(self):
        super(ConcreteDropoutLayer, self).__init__()
        self.p_logit = Parameter(torch.zeros(1))
        self.p_logit.data.fill_(-2.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        eps = 1e-8
        temp = 0.1

        p = self.sigmoid(self.p_logit)
        unif_noise = Variable(input.data.new(input.size()).uniform_())
        drop_prob = (
            torch.log((p + eps)/(1. - p + eps))
            + torch.log((unif_noise + eps)/(1. - unif_noise + eps))
        )
        random_tensor = 1. - self.sigmoid(drop_prob / temp)

        retain_prob = 1. - p
        output = input * random_tensor / (retain_prob)

        return output

class Conv1d(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, padding=0, stride=1,
                 activation='relu', batch_norm=False, dropout=0.0,
                 weight_norm=False,
    ):
        super(Conv1d, self).__init__()
        if activation in ['glu']:
            filter_size *= 2
        self.conv = nn.Conv1d(input_size, filter_size, kernel_size, stride=stride)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.bn = nn.BatchNorm1d(filter_size)
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'tanh':
            self.activ = nn.Tanh()
        if activation == 'softsign':
            self.activ = nn.Softsign()
        if activation == 'glu':
            self.activ = glu(dim=1)
        if batch_norm:
            self.batch_norm = True
        else:
            self.batch_norm = False
        if type(padding) == int:
            self.padding = (padding, padding)
        elif type(padding) == tuple:
            self.padding = padding

    def forward(self, inputs):
        inputs = F.pad(inputs, self.padding)
        outputs = self.conv(inputs)
        if self.batch_norm:
            outputs = self.bn(outputs)
        outputs = self.activ(outputs)
        if self.dropout_prob > 0:
            outputs = self.dropout(outputs)

        return outputs


class Conv2d(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, padding=0, stride=1,
                 activation='relu', batch_norm=False, dropout=0.0,
                 weight_norm=False,
    ):
        super(Conv2d, self).__init__()
        if activation in ['glu']:
            filter_size *= 2
        self.conv = nn.Conv2d(input_size, filter_size, kernel_size, stride=stride, padding=padding)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.bn = nn.BatchNorm2d(filter_size)
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'tanh':
            self.activ = nn.Tanh()
        if activation == 'softsign':
            self.activ = nn.Softsign()
        if activation == 'glu':
            self.activ = glu(dim=1)
        if batch_norm:
            self.batch_norm = True
        else:
            self.batch_norm = False

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.batch_norm:
            outputs = self.bn(outputs)
        outputs = self.activ(outputs)
        if self.dropout_prob > 0:
            outputs = self.dropout(outputs)

        return outputs
