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
import model_discriminator as md

from data import NpzFolder, NpzLoader, TBPTTIter
from utils import create_output_dir, wrap, check_grad

import training_monitor as trainmon


# from layers import ConcreteDropoutLayer, Conv2d

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


###############################
class GenderRecognitionNet(nn.Module):
    def __init__(self, seq_len, nspk):
        super(GenderRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 2d convolution
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32 * 2, 3)
        self.bn3 = nn.BatchNorm2d(32 * 2)

        self.conv4 = nn.Conv2d(32 * 2, 32 * 2, 3)
        self.bn4 = nn.BatchNorm2d(32 * 2)

        self.conv5 = nn.Conv2d(32 * 2, 32, 3)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(1 * 32 * 53, 256)  # 53 = 63 features - 5*2 for the conv filters (no padding)
        # self.fc1 = nn.Linear(1 * 32 * 43, 256)  # 53 = 63 features - 5*2 for the conv filters (no padding)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.utils.weight_norm(nn.Linear(256, nspk))
        self.fc3 = nn.Linear(256, 2)

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

        # ave pooling over time
        x = torch.max(x, dim=2)[0]

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


#####################################
class GenderRecognition(object):
    def __init__(self,
                 data_path='data/vctk',
                 checkpoint='checkpoints/gender_recognition/lastmodel.pth',
                 seq_len=300,
                 nspk=22,
                 max_seq_len=1000,
                 batch_size=64,
                 gpu=0,
                 exp_name='gender_recognition'):

        self.data_path = data_path
        self.checkpoint = checkpoint
        self.seq_len = seq_len
        self.nspk = nspk
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.gpu = gpu
        self.exp_name = exp_name

        self.train_loader, self.valid_loader = get_loaders(data_path=self.data_path,
                                                           max_seq_len=self.max_seq_len,
                                                           batch_size=self.batch_size,
                                                           nspk=self.nspk)  # TODO: add all params

        self.net = GenderRecognitionNet(seq_len=self.seq_len, nspk=self.nspk)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.speaker_info = md.get_speaker_info_for_discriminator()

        if gpu > -1:
            self.net.cuda()
            self.criterion.cuda()

        # adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)  # TODO make lr a param

        ## TODO AFTER LUNCH
        # finish coding up this object
        # test it out in a new notebook
        # save a trained model
        # reload it and evaluate on various checkpoints

    def reload_checkpoint(self, checkpoint=None):
        if checkpoint:
            self.checkpoint = checkpoint

        weights = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        # opt = torch.load(os.path.dirname(self.checkpoint) + '/args.pth')
        # train_args = opt[0]
        self.net.load_state_dict(weights)

    def evaluate(self, loader=None, epoch=1, eval_fold_str='Valid'):
        if not loader:
            loader = self.valid_loader

        total = 0
        valid_enum = tqdm(loader, desc='Eval epoch %d' % epoch)

        total_correct = 0.
        total_samples = 0.

        num_samples = len(loader.dataset)
        all_pred = []
        all_gt = []
        all_correct = []

        self.net.eval()

        for txt, feat, spkr in valid_enum:
            tmp = feat[0]
            tmp = tmp[:self.seq_len, :, :]
            feat = (tmp, feat[1])
            input = wrap(txt, volatile=True)
            target = wrap(feat, volatile=True)
            spkr = wrap(spkr, volatile=True)

            # TODO: run with gradients turned off?
            output = self.net(target[0].transpose(0, 1).unsqueeze(1))
            gender_gt = np.array(self.speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M').astype(float)
            is_male_v = Variable(torch.from_numpy(gender_gt)).long().cuda()
            gender_pred = output.cpu().data.numpy().argmax(1)

            loss = self.criterion(output, is_male_v.view(-1))

            # output, _ = model([input, spkr], target[0])
            # loss = criterion(output, target[0], target[1])

            total += loss.data[0]

            valid_enum.set_description('Evaluation %s (loss %.2f) epoch %d' %
                                       (eval_fold_str, loss.data[0], epoch))

            total_samples += len(spkr)

            # spkr_gt = spkr.cpu().view(-1).data.numpy()
            # spkr_pred = output.cpu().data.numpy().argmax(1)



            correct_pred = gender_gt == gender_pred
            num_correct_pred = np.sum(correct_pred)
            total_correct += num_correct_pred

            all_pred.append(gender_pred)
            all_gt.append(gender_gt)
            all_correct.append(correct_pred)

        accuracy = total_correct / total_samples

        avg = total / len(loader)

        all_pred = np.concatenate(all_pred)
        all_gt = np.concatenate(all_gt)
        all_correct = np.concatenate(all_correct)

        return avg, accuracy, all_pred, all_gt, all_correct

    def evaluate_synth(self, feat, spkr, epoch=1):
        tmp = feat  # [0]
        tmp = tmp[:self.seq_len, :, :]
        # feat = (tmp, feat[1])
        # target = wrap(feat, volatile=True)
        # spkr = wrap(spkr, volatile=True)

        # TODO: run with gradients turned off?
        output = self.net(tmp.transpose(0, 1).unsqueeze(1))
        num_samples = len(spkr)

        spkr_gt = spkr.cpu().view(-1).data.numpy()

        gender_gt = np.array(self.speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M')

        gender_pred = output.cpu().data.numpy().argmax(1)

        correct_pred = gender_gt == gender_pred
        num_correct_pred = np.sum(correct_pred)

        return num_correct_pred, num_samples, correct_pred, output

    def train(self, num_epochs=5):

        best_acc = 0
        train_losses = []
        self.training_monitor = trainmon.TrainingMonitor(file='gender_recognition', exp_name=self.exp_name,
                                                         b_append=True, path='training_logs',
                                                         columns=('epoch', 'update_time', 'train_loss', 'valid_loss',
                                                                  'train_acc', 'valid_acc'))

        for epoch in range(1, 1 + num_epochs):
            train_enum = tqdm(self.train_loader, desc='Train epoch %d' % epoch)

            self.net.train()

            total = 0
            for full_txt, full_feat, spkr in train_enum:
                batch_iter = TBPTTIter(full_txt, full_feat, spkr, self.max_seq_len)  # max_seq_len: will cut down later
                batch_total = 0

                counter = 1
                for txt, feat, spkr, start in batch_iter:
                    sample_lens = feat[1].numpy()
                    start_idx = np.array([np.random.randint(i + 1) for i in np.maximum(0, sample_lens - self.seq_len)])
                    tmp = feat[0].numpy()

                    x = [tmp[i:(i + self.seq_len), i, :] for i in range(len(start_idx))]
                    y = np.array(x)
                    y = y.transpose(1, 0, 2)

                    feat = (torch.FloatTensor(y), feat[1])

                    input = wrap(txt)  # volatile=True if we want to test with less memory
                    target = wrap(feat)
                    spkr = wrap(spkr)

                    is_male = np.array(self.speaker_info.iloc[spkr.cpu().data.numpy().flatten()].gender == 'M').astype(
                        float)
                    is_male_v = Variable(torch.from_numpy(is_male)).long().cuda()
                    # print type(is_male_v)

                    # Zero gradients
                    if start:
                        self.optimizer.zero_grad()

                    # Forward
                    output = self.net(target[0].transpose(0, 1).unsqueeze(1))
                    loss = self.criterion(output, is_male_v.view(-1))

                    # Backward
                    loss.backward()
                    # if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                    #    logging.info('Not a finite gradient or too big, ignoring.')
                    #    optimizer.zero_grad()
                    #    continue
                    self.optimizer.step()

                    # Keep track of loss
                    batch_total += loss.data[0]
                    counter += 1

                batch_total = batch_total / len(batch_iter)
                total += batch_total
                train_enum.set_description('Train (loss %.3f) epoch %d' %
                                           (batch_total, epoch))

            avg = total / len(self.train_loader)
            train_losses.append(avg)

            train_loss, train_accuracy, all_pred, all_gt, all_correct = self.evaluate(self.train_loader, epoch=epoch,
                                                                                      eval_fold_str='train')
            print "Training accuracy (epoch %d): %.3f" % (epoch, train_accuracy)

            valid_loss, valid_accuracy, all_pred, all_gt, all_correct = self.evaluate(self.valid_loader, epoch=epoch)
            print "Validation accuracy (epoch %d): %.3f" % (epoch, valid_accuracy)

            self.training_monitor.insert(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss,
                                         train_acc=train_accuracy, valid_acc=valid_accuracy)

            if valid_accuracy > best_acc:
                best_acc = valid_accuracy

                exp_name = os.path.join('checkpoints', self.exp_name)
                if not os.path.exists(exp_name):
                    os.makedirs(exp_name)

                torch.save(self.net.state_dict(), '%s/bestmodel.pth' % (exp_name))
                # torch.save([train_losses, eval_dict],
                #           '%s/args.pth' % (exp_name))

        # all training done. Build final loss & accuracy stats
        train_loss, train_accuracy, all_pred, all_gt, all_correct = self.evaluate(self.train_loader)
        valid_loss, valid_accuracy, all_pred, all_gt, all_correct = self.evaluate(self.valid_loader)

        # store summary states in a dict
        eval_dict = dict()
        eval_dict['train_loss'] = train_loss
        eval_dict['train_accuracy'] = train_accuracy
        eval_dict['valid_loss'] = valid_loss
        eval_dict['valid_accuracy'] = valid_accuracy
        eval_dict['valid_pred'] = all_pred
        eval_dict['valid_gt'] = all_gt
        eval_dict['valid_correct'] = all_correct

        self.train_eval_dict = eval_dict
        return eval_dict


#################################


def train_gender_recognition(gpu=0,
                             seed=1,
                             data_path='/home/ubuntu/loop/data/vctk',
                             nspk=22,
                             max_seq_len=1000,
                             seq_len=300,
                             batch_size=64,
                             num_epochs=5,
                             exp_name='gender_recognition',
                             checkpoint=None):
    torch.manual_seed(seed)
    if gpu > -1:
        torch.cuda.set_device(gpu)
        torch.cuda.manual_seed(seed)

    sr = GenderRecognition(data_path=data_path,
                           checkpoint=checkpoint,
                           seq_len=seq_len,
                           nspk=nspk,
                           max_seq_len=max_seq_len,
                           batch_size=batch_size,
                           gpu=gpu,
                           exp_name=exp_name)

    eval_dict = sr.train(num_epochs=num_epochs)

    exp_name = os.path.join('checkpoints', exp_name)
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    torch.save(sr.net.state_dict(), '%s/lastmodel.pth' % (exp_name))
    torch.save([eval_dict],
               '%s/args.pth' % (exp_name))

    return sr


def main(exp_name='gender_recognition', num_epochs=2):
    sr = train_gender_recognition(num_epochs=num_epochs)


if __name__ == '__main__':
    main()
