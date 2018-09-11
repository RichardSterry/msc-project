import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import numpy as np
import pandas as pd
from notebook_utils import get_vctk_speaker_info
from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
import evaluate_loss_func_for_notebook as el
import copy


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, y=None):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b

#if weight_norm:
#    in_layer = nn.utils.weight_norm(in_layer)
#    same out layer

class LatentDiscriminator(nn.Module):
    def __init__(self, n_in=256):
        super(LatentDiscriminator, self).__init__()
        self.n_in = n_in
        self.fc1 = nn.utils.weight_norm(nn.Linear(n_in, 32))
        self.fc2 = nn.utils.weight_norm(nn.Linear(32, 16))
        self.fc3 = nn.utils.weight_norm(nn.Linear(16, 2))

    def reset(self):
        self.fc1 = nn.Linear(self.n_in, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def cuda(self, device_id=None):
        nn.Module.cuda(self, device_id)


def evaluate_latent_discriminator(net, data, criterion):

    num_samples = len(data[0])

    x = data[0]
    y = data[1]

    x_wrap = Variable(torch.from_numpy(x)).cuda()

    y_wrap = Variable(torch.from_numpy(y.astype('uint8')).type(torch.FloatTensor)).cuda()

    output = net(x_wrap)

    loss = criterion(output, y_wrap.type(torch.cuda.LongTensor))

    y_pred = output.cpu().data.numpy().argmax(axis=1)
    correct_pred = y == y_pred
    num_correct_pred = np.sum(correct_pred)

    accuracy = 1.0*num_correct_pred / num_samples

    return accuracy, loss.cpu().data.numpy()[0]


def train_discriminator(net, train_data, valid_data, criterion, optimizer, num_epochs=500, b_print=True):
    net.train()

    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)
    train_accuracy = np.zeros(num_epochs)
    valid_accuracy = np.zeros(num_epochs)

    #print "Initialized: train %0.3f / validation %0.3f" % (evaluate_latent_discriminator(net, train_data, criterion)[0], evaluate_latent_discriminator(net, valid_data, criterion)[0])

    for epoch in range(num_epochs):
        num_samples = len(train_data[0])
        all_pred = []
        all_gt = []
        all_correct = []

        x = train_data[0]
        y = train_data[1]

        x_wrap = Variable(torch.from_numpy(x)).cuda()

        y_wrap = Variable(torch.from_numpy(y.astype('uint8')).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()

        # Forward
        output = net(x_wrap)

        loss = criterion(output, y_wrap.type(torch.cuda.LongTensor))

        # Backward
        loss.backward()

        optimizer.step()

        # print loss.cpu().data.numpy()[0]
        this_train_accuracy, this_train_loss = evaluate_latent_discriminator(net, train_data, criterion)
        this_valid_accuracy, this_valid_loss = evaluate_latent_discriminator(net, valid_data, criterion)
        train_accuracy[epoch] = this_train_accuracy
        valid_accuracy[epoch] = this_valid_accuracy
        train_loss[epoch] = this_train_loss
        valid_loss[epoch] = this_valid_loss
        #print "Epoch %d: loss %0.6f, train %0.3f / validation %0.3f" % (
        #epoch, loss.cpu().data.numpy()[0], this_train_accuracy, this_valid_accuracy)


        # avg = total / len(train_loader)
        # train_losses.append(avg)


        # logging.info('====> Train set loss: {:.4f}'.format(avg))

    if b_print:
        print "Discriminator: loss %0.6f, train %0.3f / validation %0.3f" % (loss.cpu().data.numpy()[0], this_train_accuracy, this_valid_accuracy)

    return net, train_accuracy, valid_accuracy, train_loss, valid_loss


def eval_discriminator(net, train_data, criterion):

    #print "Initialized: train %0.3f / validation %0.3f" % (evaluate_latent_discriminator(net, train_data, criterion)[0], evaluate_latent_discriminator(net, valid_data, criterion)[0])

    # taken out 18-Jul in the evening - not sure if this is correct
    #net.eval()

    x = train_data[0]
    y = train_data[1]

    y = (y + 1) % 2 # flip correct class - not sure where the idea for this came from, need to find out...

    #x_wrap = Variable(torch.from_numpy(x)).cuda()
    x_wrap = x
    y_wrap = Variable(torch.from_numpy(y.astype('uint8')).type(torch.FloatTensor)).cuda()

    # Forward
    output = net(x_wrap)

    loss = criterion(output, y_wrap.type(torch.cuda.LongTensor))

    return loss


def eval_discriminator_accuracy(net, train_data, criterion):

    net.eval()

    num_samples = len(train_data[0])

    x = train_data[0]
    y = train_data[1]

    if isinstance(x, np.ndarray):
        x_wrap = Variable(torch.from_numpy(x)).cuda()
    else:
        x_wrap = x

   #y_wrap = Variable(torch.from_numpy(y.astype('uint8')).type(torch.FloatTensor)).cuda()

    # Forward
    output = net(x_wrap)

    y_pred = output.cpu().data.numpy().argmax(axis=1)
    correct_pred = y == y_pred
    num_correct_pred = np.sum(correct_pred)

    accuracy = 1.0 * num_correct_pred / num_samples

    return accuracy


def get_speaker_info_for_discriminator():
    # location of the float32, train/validation files for VCTK-all
    vctk_precalc_folder = '/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all'

    speaker_info = get_vctk_speaker_info()

    loader = el.get_loader(data_path=vctk_precalc_folder)

    loop_speaker_lookup = loader.dataset.speakers

    # Create dict from IDs used inside VoiceLoop to VCTK speaker IDs
    speaker_list_vctk = [int(k[1:]) for k in loop_speaker_lookup.keys()]  # list of VCTK speaker IDs; strip out the 'p'
    speaker_dict_vctk_to_loop = dict(zip(speaker_list_vctk, loop_speaker_lookup.values()))  # dict['vctk_id] = sim_id
    speaker_dict_loop_to_vctk = dict(zip(loop_speaker_lookup.values(), speaker_list_vctk))  # dict['sim_id] = vctk_id

    # get dataframe for VCTK reference data indexed by VoiceLoop speaker ID
    tmp = pd.DataFrame.from_dict(speaker_dict_loop_to_vctk, orient='index', columns=['id'])
    speaker_info_loop = pd.merge(speaker_info, tmp)

    return speaker_info_loop


def get_speaker_embeddings():
    # model checkpoint
    checkpoint_file = '/home/ubuntu/loop/checkpoints/vctk-16khz-cmu-no-boundaries-all-noise-2/bestmodel.pth'

    checkpoint_args_path = os.path.join(os.path.dirname(checkpoint_file), 'args.pth')
    checkpoint_args = torch.load(checkpoint_args_path)

    # restore the model from the checkpoint
    checkpoint = torch.load(checkpoint_file,
                            map_location=lambda storage, loc: storage)

    model = Loop(checkpoint_args[0])

    model.load_state_dict(checkpoint)

    # extract the speaker embeddings from the model
    embeddings = model.encoder.lut_s.weight.data.numpy()

    if embeddings.shape[0] == 108:
        embeddings = np.delete(embeddings, -1, axis=0)

    return embeddings


def get_train_valid_split(embeddings, speaker_info, cutoff=None):

    num_examples = embeddings.shape[0]

    b_male = np.array(speaker_info.gender == 'M')

    if cutoff:
        # split the speakers into train and validation sets
        idx_rand = np.random.permutation(num_examples)
        idx_train = idx_rand[:cutoff]
        idx_valid = idx_rand[cutoff:]
        train_data = (embeddings[idx_train], b_male[idx_train])
        valid_data = (embeddings[idx_valid], b_male[idx_valid])
    else:
        # just return the whole set
        train_data = (embeddings, b_male)
        valid_data = train_data

    return train_data, valid_data


def build_discriminator(gpu=0, seed=1):
    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net = LatentDiscriminator()
    net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    return net, criterion, optimizer
