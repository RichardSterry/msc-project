import os
import sys

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from data import NpzFolder, NpzLoader, TBPTTIter
from utils import create_output_dir, wrap, check_grad



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
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 square convolution
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, 3)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, 3)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(1 * 32 * 53, 256) # 53 = 63 features - 5*2 for the conv filters (no padding)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, nspk)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # print x.size()
        # print type(x)
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = F.relu(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # print x.size()
        x = F.relu(self.bn2(self.conv2(x)))
        # print x.size()
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # print x.size()

        # ave pooling over time
        # x = self.pool(x)
        x = torch.max(x, dim=2)[0]
        # print x.size()

        x = x.view(-1, self.num_flat_features(x))
        # print x.size()
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

    for txt, feat, spkr in valid_enum:
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




def train(data_path='data/vctk', seq_len=300, nspk=22, num_epochs=5, batch_size=64, max_seq_len=1000):

    # get dataset loaders
    train_loader, valid_loader = get_loaders(data_path, max_seq_len, batch_size)

    # speaker recognition net
    net = RecognitionNet(seq_len=seq_len, nspk=nspk)
    net.cuda()
    net.train()

    # cross entropy loss
    criterion = nn.CrossEntropyLoss().cuda()

    # adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(num_epochs):
        train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

        total = 0
        for full_txt, full_feat, spkr in train_enum:
            batch_iter = TBPTTIter(full_txt, full_feat, spkr, seq_len)
            batch_total = 0

            counter = 1
            for txt, feat, spkr, start in batch_iter:
                input = wrap(txt)
                target = wrap(feat)
                spkr = wrap(spkr)

                # Zero gradients
                if start:
                    optimizer.zero_grad()

                # Forward
                #output = net(target[0].transpose(0,1).unsqueeze(1).cpu())
                output = net(target[0].transpose(0,1).unsqueeze(1))
                #print output.size()
                #print spkr.size()
                loss = criterion(output, spkr.view(-1))
                #print "Iteration %d: loss %.2f" %(counter, loss.data[0])

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

                break

            batch_total = batch_total / len(batch_iter)
            total += batch_total
            train_enum.set_description('Train (loss %.3f) epoch %d' %
                                       (batch_total, epoch))
            #print "Train (loss %.2f)" % (batch_total)

            #break

        avg = total / len(train_loader)
        train_losses.append(avg)

        valid_loss, valid_accuracy, all_pred, all_gt, all_correct = evaluate(net, criterion, valid_loader)
        print "Validation accuracy: %.3f" % valid_accuracy

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
                                num_epochs = 5):

    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net, criterion, train_losses, eval_dict = train(data_path=data_path,
                                                    seq_len=seq_len,
                                                    nspk=nspk,
                                                    num_epochs=num_epochs,
                                                    max_seq_len=max_seq_len,
                                                    batch_size=batch_size)

    return net, criterion, train_losses, eval_dict



def main(exp_name = 'speaker_recognition'):
    net, criterion, train_losses, eval_dict = train_speaker_recognition(num_epochs=1)

    exp_name = os.path.join('checkpoints', exp_name)
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    torch.save(net.state_dict(), '%s/lastmodel.pth' % (exp_name))
    torch.save([train_losses, eval_dict],
               '%s/args.pth' % (exp_name))


if __name__ == '__main__':
    main()
