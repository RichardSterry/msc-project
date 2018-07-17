import os
import sys

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

from data import NpzFolder, NpzLoader, TBPTTIter
from utils import create_output_dir, wrap, check_grad

#from layers import ConcreteDropoutLayer, Conv2d

class Conv2d(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, padding=0, stride=1,
                 activation='relu', batch_norm=False, dropout=0.0,
                 weight_norm=False,
    ):
        #super().__init__()
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

def get_loaders(data_path='data/vctk', max_seq_len=1000, batch_size=64, nspk=22):
    # wrap train dataset
    train_dataset = NpzFolder(data_path + '/numpy_features', nspk == 1)
    train_loader = NpzLoader(train_dataset,
                             max_seq_len=max_seq_len,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=True)

    # wrap validation dataset
    valid_dataset = NpzFolder(data_path + '/numpy_features_valid', nspk == 1)
    valid_loader = NpzLoader(valid_dataset,
                             max_seq_len=max_seq_len,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, valid_loader


class RecognitionNet(nn.Module):
    def __init__(self, seq_len, nspk):
        super(RecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 2d convolution
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32*2, 3)
        self.bn3 = nn.BatchNorm2d(32*2)

        self.conv4 = nn.Conv2d(32*2, 32*2, 3)
        self.bn4 = nn.BatchNorm2d(32*2)

        self.conv5 = nn.Conv2d(32*2, 32, 3)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(1 * 32 * 53, 256) # 53 = 63 features - 5*2 for the conv filters (no padding)
        #self.fc1 = nn.Linear(1 * 32 * 43, 256)  # 53 = 63 features - 5*2 for the conv filters (no padding)
        self.fc2 = nn.Linear(256, 256)
        #self.fc3 = nn.utils.weight_norm(nn.Linear(256, nspk))
        self.fc3 = nn.Linear(256, nspk)

        # take out bias term
        # try strides
        # make filter biggers

        self.drop = nn.Dropout2d(0.1)
        self.fully_connected = 64
        self.gru = torch.nn.GRU(544, self.fully_connected, 1, bidirectional=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # [B, C, H, T] -> [B, C*H, T]
        #out = out.view(out.size(0),
        #               -1,
        #               out.size(3))

        # [B, C, T] -> [T, B, C]
        #out = out.permute(2, 0, 1).contiguous()

        #RNN = True
        ## [T, B, C]
        #if RNN:
        #    if start:
        #        h_0 = self.h_0.expand(self.h_0.size(0),
        #                              out.size(1),
        #                              self.fully_connected)
        #    else:
        #        h_0 = Variable(self.h_n.data.new(self.h_n.size()))
        #        h_0.data = self.h_n.data.clone()
        #    h_0 = h_0.contiguous()
        #    out, self.h_n = self.gru(out, h_0)


        # ave pooling over time
        # x = self.pool(x)
        x = torch.max(x, dim=2)[0]
        #x = torch.mean(x, dim=2)
        # print x.size()

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def cuda(self, device_id=None):
        nn.Module.cuda(self, device_id)


class RecognitionNetJiameng(nn.Module):
    def __init__(self, nspk=108, embedding_size=256, hidden_size=256, activation ='relu'):
        super(RecognitionNetJiameng, self).__init__()
        self.input_size = 67
        self.embedding_size = embedding_size
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
        self.gru = torch.nn.GRU(544, self.fully_connected, 1, bidirectional=True)
        self.h_0 = Parameter(torch.zeros(2, 1, self.fully_connected))

        self.fc_out = nn.utils.weight_norm(nn.Linear(self.fully_connected * 2, self.embedding_size))

        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softsign = nn.Softsign()


        self.spkr_num = nspk
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
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

        eps = Variable(x.data.new(x.size()).normal_(0, 0.01))
        x = x + eps

        # [T, B, C] -> [T, 1, B, C]
        #out = x.unsqueeze(1)
        out = x

        # [T, 1, B, C] -> [B, 1, C, T]
        out = out.permute(2, 1, 3, 0)
        out.contiguous()

        # Randomize input
        #out = F.pad(out, (20, 0, 0, 0), mode='replicate')
        #rnd = np.random.randint(20)
        #lengths = lengths + (20 - rnd)
        #out = out[:, :, :, rnd:]

        # [B, C, H, T]
        for i, conv in enumerate(self.conv):
            out = self.dropout[i](out)
            out = conv(out)

        # [B, C, H, T] -> [B, C*H, T]
        out = out.view(out.size(0),
                       -1,
                       out.size(3))

        # [B, C, T] -> [T, B, C]
        out = out.permute(2, 0, 1).contiguous()

        RNN = True
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

        def transform(lengths):
            return ((lengths.float() + 2 - 2 - 1) / 2 + 1).floor()

        lengths = transform(lengths)
        lengths = transform(lengths)
        lengths = transform(lengths)

        mask = self._sequence_mask(lengths.long()).unsqueeze(2)
        mask_ = mask.expand_as(out)

        # [T, B, C] -> [B, C]
        time_norm = 'mean'
        if time_norm == 'mean':
            # out = torch.sum(out*mask_.float(), dim=0) / mask_.float().sum(0)
            out = torch.mean(out, dim=0)
        elif time_norm == 'last':
            out = out[-1, :, :]

        ident_s = torch.tanh(self.fc_out(out))

        ident_s = torch.renorm(ident_s, 2, dim=0, maxnorm=1.0)

        out = self.fc_in(ident_s)
        out = self.activ(out)
        for i, layer in enumerate(self.fc_hidden):
            out = layer(out)
        out = self.fc_out(out)
        return out

    def cuda(self, device_id=None):
        nn.Module.cuda(self, device_id)



class SpeakerRecognition(object):
    def __init__(self, checkpoint='checkpoints/speaker_recognition/lastmodel.pth', ):
        weights = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')
        train_args = opt[0]

        self.net = RecognitionNet()


def evaluate(net, criterion, loader, epoch=1, eval_losses=[]):
    total = 0
    valid_enum = tqdm(loader, desc='Valid epoch %d' % epoch)
    # valid_enum = loader

    total_correct = 0.
    total_samples = 0.

    num_samples = len(loader.dataset)
    all_pred = []
    all_gt = []
    all_correct = []

    net.eval()

    for txt, feat, spkr in valid_enum:
        tmp = feat[0]
        tmp = tmp[:300, :, :]
        feat = (tmp, feat[1])
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        # TODO: run with gradients turned off?
        output = net(target[0].transpose(0, 1).unsqueeze(1))
        loss = criterion(output, spkr.view(-1))

        # output, _ = model([input, spkr], target[0])
        # loss = criterion(output, target[0], target[1])

        total += loss.data[0]

        valid_enum.set_description('Evaluation (loss %.2f) epoch %d' %
                                   (loss.data[0], epoch))

        total_samples += len(spkr)

        spkr_gt = spkr.cpu().view(-1).data.numpy()
        spkr_pred = output.cpu().data.numpy().argmax(1)

        correct_pred = spkr_gt == spkr_pred
        num_correct_pred = np.sum(correct_pred)
        total_correct += num_correct_pred

        all_pred.append(spkr_pred)
        all_gt.append(spkr_gt)
        all_correct.append(correct_pred)

    accuracy = total_correct / total_samples

    avg = total / len(loader)
    eval_losses.append(avg)

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    all_correct = np.concatenate(all_correct)

    return avg, accuracy, all_pred, all_gt, all_correct


def train(data_path='data/vctk', seq_len=300, nspk=22, num_epochs=5, batch_size=64, max_seq_len=1000, exp_name='speaker_recognition', gpu=0):

    # get dataset loaders
    train_loader, valid_loader = get_loaders(data_path, max_seq_len, batch_size)

    # speaker recognition net
    net = RecognitionNet(seq_len=seq_len, nspk=nspk)
    #net = RecognitionNetJiameng(nspk=nspk, hidden_size=256, embedding_size=256)
    if gpu > -1:
        net.cuda()

    net.train()

    # cross entropy loss
    if gpu > -1:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    best_acc = 0
    train_losses = []
    for epoch in range(num_epochs):
        train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

        net.train()

        total = 0
        for full_txt, full_feat, spkr in train_enum:
            batch_iter = TBPTTIter(full_txt, full_feat, spkr, max_seq_len) # max_seq_len: will cut down later
            batch_total = 0

            counter = 1
            for txt, feat, spkr, start in batch_iter:
                sample_lens = feat[1].numpy()
                start_idx = np.array([np.random.randint(i + 1) for i in np.maximum(0, sample_lens - seq_len)])
                tmp = feat[0].numpy()

                x = [tmp[i:(i + seq_len), i, :] for i in range(len(start_idx))]
                y = np.array(x)
                y = y.transpose(1, 0, 2)

                feat = (torch.FloatTensor(y), feat[1])

                input = wrap(txt) # volatile=True if we want to test with less memory
                target = wrap(feat)
                spkr = wrap(spkr)

                # Zero gradients
                if start:
                    optimizer.zero_grad()

                # Forward
                output = net(target[0].transpose(0,1).unsqueeze(1))
                loss = criterion(output, spkr.view(-1))

                # Backward
                loss.backward()
                #if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                #    logging.info('Not a finite gradient or too big, ignoring.')
                #    optimizer.zero_grad()
                #    continue
                optimizer.step()


                # Keep track of loss
                batch_total += loss.data[0]
                counter += 1

            batch_total = batch_total / len(batch_iter)
            total += batch_total
            train_enum.set_description('Train (loss %.3f) epoch %d' %
                                       (batch_total, epoch))


        avg = total / len(train_loader)
        train_losses.append(avg)

        train_loss, train_accuracy, all_pred, all_gt, all_correct = evaluate(net, criterion, train_loader)
        print "Training accuracy: %.3f" % train_accuracy

        valid_loss, valid_accuracy, all_pred, all_gt, all_correct = evaluate(net, criterion, valid_loader)
        print "Validation accuracy: %.3f" % valid_accuracy

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy

            exp_name = os.path.join('checkpoints', exp_name)
            if not os.path.exists(exp_name):
                os.makedirs(exp_name)

            torch.save(net.state_dict(), '%s/bestmodel.pth' % (exp_name))
            #torch.save([train_losses, eval_dict],
            #           '%s/args.pth' % (exp_name))


    # all training done. Build final loss & accuracy stats
    train_loss, train_accuracy, all_pred, all_gt, all_correct = evaluate(net, criterion, train_loader, epoch=epoch)
    valid_loss, valid_accuracy, all_pred, all_gt, all_correct = evaluate(net, criterion, valid_loader, epoch=epoch)

    # store summary states in a dict
    eval_dict = dict()
    eval_dict['train_loss'] = train_loss
    eval_dict['train_accuracy'] = train_accuracy
    eval_dict['valid_loss'] = valid_loss
    eval_dict['valid_accuracy'] = valid_accuracy
    eval_dict['valid_pred'] = all_pred
    eval_dict['valid_gt'] = all_gt
    eval_dict['valid_correct'] = all_correct

    return net, criterion, train_losses, eval_dict


def train_speaker_recognition(  gpu=0,
                                seed=1,
                                data_path = '/home/ubuntu/loop/data/vctk',
                                nspk = 22,
                                max_seq_len = 1000,
                                seq_len = 300,
                                batch_size = 64,
                                num_epochs = 5,
                                exp_name='speaker_recognition'):
    torch.manual_seed(seed)
    if gpu>-1:
        torch.cuda.set_device(gpu)
        torch.cuda.manual_seed(seed)

    net, criterion, train_losses, eval_dict = train(data_path=data_path,
                                                    seq_len=seq_len,
                                                    nspk=nspk,
                                                    num_epochs=num_epochs,
                                                    max_seq_len=max_seq_len,
                                                    batch_size=batch_size,
                                                    exp_name=exp_name,
                                                    gpu=gpu)

    exp_name = os.path.join('checkpoints', exp_name)
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    torch.save(net.state_dict(), '%s/lastmodel.pth' % (exp_name))
    torch.save([train_losses, eval_dict],
               '%s/args.pth' % (exp_name))

    return net, criterion, train_losses, eval_dict



def main(exp_name = 'speaker_recognition'):
    net, criterion, train_losses, eval_dict = train_speaker_recognition(num_epochs=1, gpu=-1)




if __name__ == '__main__':
    main()
