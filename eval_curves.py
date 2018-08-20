# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from tqdm import tqdm

import torch

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
from utils import create_output_dir, wrap, check_grad
from training_monitor import TrainingMonitor
import pickle
import speaker_recognition as sr
from dtw import dtw
from torch.autograd import Variable

logSpecDbConst = 10.0 / np.log(10.0) * np.sqrt(2.0)


def logSpecDbDist(x, y):
    diff = x - y
    return logSpecDbConst * np.sqrt(np.inner(diff, diff))


def get_loader(data_path='data/vctk', max_seq_len=1000, batch_size=64, nspk=22, b_valid=True):
    if b_valid:
        dir = '/numpy_features_valid'
        shuffle = False
    else:
        dir = '/numpy_features'
        shuffle = True

    dataset = NpzFolder(data_path + dir, nspk == 1)

    loader = NpzLoader(dataset,
                             max_seq_len=max_seq_len,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=shuffle)
    return loader


def get_training_data_for_eval(data, len_valid=None, max_seq_len=1000, batch_size=64):
    """random set of training examples of the same size as the validation set"""
    #if not len_valid:
    #    len_valid = len(valid_dataset)

    # get list of all samples in the training fold
    train_eval_dataset = NpzFolder(data + '/numpy_features', False)

    # pick a random subset with the same number of samples as the validation fold
    idx = np.random.permutation(len(train_eval_dataset))[:len_valid]
    train_eval_dataset.npzs = [train_eval_dataset.npzs[i] for i in idx]

    # loader for this train_eval dataset
    train_eval_loader = NpzLoader(train_eval_dataset,
                             max_seq_len=max_seq_len,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=True)

    return train_eval_loader


def load_checkpoint(checkpoint='models/vctk/bestmodel.pth', data='data/vctk', max_seq_len=1000, nspk=22, gpu=0, batch_size=64, seed=1):

    checkpoint_args_path = os.path.dirname(checkpoint) + '/args.pth'
    checkpoint_args = torch.load(checkpoint_args_path)

    opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')
    train_args = opt[0]
    #train_args.noise = 0
    train_args.checkpoint = checkpoint

    model = Loop(train_args)

    model.cuda()
    model.load_state_dict(torch.load(train_args.checkpoint, map_location=lambda storage, loc: storage))
    #model.eval()

    return model


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


def evaluate(model, criterion, epoch, loader, speaker_recog=None,
             metrics=None, kld_lambda=1.):

    if not metrics:
        metrics = ('loss', 'mcd', 'loss_workings', 'speaker_recognition')

    total_mse = 0
    total_kld = 0
    total = 0
    my_total = 0
    loss = 0
    my_loss = 0
    my_avg = 0
    loss_contrib = []
    loss_workings = None

    num_samples = len(loader.dataset)
    sr_all_pred = []
    sr_all_gt = []
    sr_all_correct = []
    sr_total_correct = 0.
    sr_total_samples = 0

    all_output = []
    all_txt = []
    all_txt_len = []
    all_target_feat = []
    all_target_len = []
    all_spkr = []
    all_loss_contrib = []
    all_attn = []

    mcd_total = 0
    mcd_avg = None
    dtw_cost = logSpecDbDist

    norm = np.load(loader.dataset.npzs[0])['audio_norminfo']
    cmp_mean = norm[0]
    cmp_std = norm[1]

    sr_accuracy = None

    valid_enum = tqdm(loader, desc='Valid epoch %d' % epoch)

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        # !!!!!!!! temporary hack !!!!!!!
        # switch this back on when doing speaker recognition evaluation
        if True:
            tmp = target[0]#.permute(0, 2, 1)
            ident_mu, ident_logvar = model.get_embeddings(tmp, start=True)
            model.train()  # !!temp
            embedding_array = model.reparameterize(ident_mu, ident_logvar)
            model.eval()  # !!temp
            output, _, _, ident_mu, ident_logvar = model([input, spkr], target[0], embedding_array=embedding_array)
            model.train()
        else:
            # loop forward pass
            output, _, _, ident_mu, ident_logvar = model([input, spkr], target[0])

        # loss calculation
        if 'loss' in metrics:
            loss_mse = criterion(output, target[0], target[1])
            #kld = -0.5 * torch.sum(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())
            kld = -0.5 * torch.mean(1 + ident_logvar - ident_mu.pow(2) - ident_logvar.exp())
            loss = loss_mse + kld_lambda * kld

            total_mse += loss_mse.data[0]
            total_kld += kld.data[0]
            total += loss.data[0]

        # speaker recognition on synth samples
        if 'speaker_recognition' in metrics:
            num_correct_pred, num_samples, correct_pred = speaker_recog.evaluate_synth(feat=output, spkr=spkr)

            sr_total_correct += num_correct_pred
            sr_total_samples += len(spkr)

            # all_pred.append(spkr_pred)
            # all_gt.append(spkr_gt)
            sr_all_correct.append(correct_pred)


        # mcd calc
        if 'mcd' in metrics:
            batch_size = output.size(1)
            tmp_mcd_loss = 0

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
                tmp_mcd_loss += unit_loss

            tmp_mcd_loss /= batch_size

            mcd_total += tmp_mcd_loss


        # manual MSE calc (for loss analysis)
        if 'loss_workings' in metrics:
            my_loss, loss_contrib = mse_manual(output, target[0], target[1])
            my_total += my_loss.data[0]
            all_loss_contrib.append(loss_contrib.cpu().data.numpy() / len(loader))  # divide by number of batches

            all_output.append(output.cpu().data.numpy())
            all_txt.append(txt[0].cpu().numpy())
            all_txt_len.append(txt[1].cpu().numpy())
            all_target_feat.append(target[0].cpu().data.numpy())
            all_target_len.append(target[1].cpu().data.numpy())
            all_spkr.append(spkr.cpu().data.numpy().squeeze())

            #all_attn.append(attn.cpu().data.numpy())
            #valid_enum.set_description('Loss %.2f' % (loss.data[0]))


        # update logger for this mini batch
        valid_enum.set_description('Valid (loss %.2f) epoch %d' %
                                   (loss.data[0], epoch))


    # total loss across batch
    loss_avg_mse = total_mse / len(loader)
    loss_avg_kld = total_kld / len(loader)
    loss_avg = total / len(loader)

    # total speaker recognition accuracy across batch
    if 'speaker_recognition' in metrics:
        sr_accuracy = sr_total_correct / sr_total_samples
        #sr_all_pred = np.concatenate(sr_all_pred)
        #sr_all_gt = np.concatenate(sr_all_gt)
        sr_all_correct = np.concatenate(sr_all_correct)

    # total mcd across batch
    if 'mcd' in metrics:
        mcd_avg = mcd_total / len(loader)

    # loss workings output
    if 'loss_workings' in metrics:
        # manual MSE output
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

    return loss_avg, loss_avg_mse, loss_avg_kld, sr_accuracy, mcd_avg, loss_workings


def calc_eval_curves(checkpoint_folder='models/vctk-all/',
                     checkpoint_file='bestmodel.pth',
                     start_epoch=None,
                     step_epoch=1,
                     end_epoch=None,
                     eval_epochs=10,
                     data='data/vctk',
                     max_seq_len=1000,
                     nspk=22,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     b_append=False,
                     exp_name='test',
                     speaker_recognition_checkpoint='checkpoints/speaker_recognition_vctk_all/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     b_teacher_force=True,
                     b_use_train_noise=True,
                     eval_metrics=('loss', 'speaker_recognition')):

    assert os.path.isdir(checkpoint_folder), "checkpoint_folder needs to be a valid directory"

    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader = get_loader(data, max_seq_len, batch_size, nspk, b_valid=False)
    valid_loader = get_loader(data, max_seq_len, batch_size, nspk, b_valid=True)

    criterion = MaskedMSE().cuda()

    # Keep track of losses
    training_monitor = TrainingMonitor(file=exp_name, exp_name=exp_name, b_append=b_append, path='training_logs',
                                       columns=('epoch', 'update_time', 'train_loss', 'valid_loss', 'mcd', 'speaker_recognition_acc_eval'))

    # Load pre-trained SpeakerRecognition model
    speaker_recog = sr.SpeakerRecognition(data_path=data,
                                               checkpoint=speaker_recognition_checkpoint,
                                               seq_len=300,
                                               nspk=nspk,
                                               max_seq_len=max_seq_len,
                                               batch_size=batch_size,
                                               gpu=gpu,
                                               exp_name=speaker_recognition_exp_name)

    speaker_recog.reload_checkpoint()

    # Prepare list of checkpoints to evaluate
    if start_epoch:
        # if start_epoch is specified, evaluate checkpoints over a range of epochs
        epoch_list = range(start_epoch, end_epoch + 1, step_epoch)
        checkpoint_list = [os.path.join(checkpoint_folder, "epoch_" + str(e) + ".pth") for e in epoch_list]
    else:
        # ...else evaluate a single, named checkpoint file
        this_checkpoint_file = os.path.join(checkpoint_folder, checkpoint_file)
        checkpoint_list = (this_checkpoint_file, )
        epoch_list = (1,)


    # Loop through & evaluate the checkpoints
    for epoch, this_checkpoint_file in zip(epoch_list, checkpoint_list):
        # load model from checkpoint
        model = load_checkpoint(this_checkpoint_file)

        # set noise & teacher forcing
        if b_teacher_force:
            model.train()
        else:
            model.eval()

        if not b_use_train_noise:
            model.noise = 0

        # core evaluation metrics
        if epoch % eval_epochs == 0:
            this_epoch_metrics = None
        else:
            this_epoch_metrics = eval_metrics
        #loss_avg, loss_avg_mse, loss_avg_kld, sr_accuracy, mcd_avg, loss_workings
        valid_loss, loss_avg_mse, loss_avg_kld, valid_sr_accuracy, valid_mcd_avg, valid_loss_workings = evaluate(model, criterion=criterion, epoch=epoch,
                                                                                     loader=valid_loader, speaker_recog=speaker_recog,
                                                                                     metrics=this_epoch_metrics)
        save_loss_workings(exp_name, epoch, valid_loss_workings)

        train_eval_loader = get_training_data_for_eval(data=data, len_valid=len(valid_loader.dataset))
        train_loss, loss_avg_mse, loss_avg_kld, train_sr_accuracy, train_mcd_avg, train_loss_workings = evaluate(model, criterion=criterion, epoch=epoch,
                                                                  loader=train_eval_loader,
                                                                  speaker_recog=speaker_recog,
                                                                  metrics = ('loss'))

        training_monitor.insert(epoch=epoch,
                                valid_loss=valid_loss,
                                train_loss=train_loss,
                                mcd=valid_mcd_avg,
                                speaker_recognition_acc_eval=valid_sr_accuracy)
        training_monitor.write()

    training_monitor.disp()
    return training_monitor


def save_loss_workings(exp_name, epoch, loss_workings):
    if loss_workings:
        loss_file = os.path.join('training_logs', exp_name + '_loss_contrib_' + str(epoch) + '.pickle')
        with open(loss_file, 'wb') as handle:
            pickle.dump(loss_workings['loss_contrib'], handle, protocol=pickle.HIGHEST_PROTOCOL)





#################################
if __name__ == '__main__':
    #calc_eval_curves(checkpoint_folder='checkpoints/vctk-us-vae-64-mean-lambda-0.5',
    #                 data='/home/ubuntu/loop/data/vctk',
    #                 speaker_recognition_checkpoint='/home/ubuntu/msc-project-master/msc-project-master/checkpoints/speaker-recognition-vctk-us/bestmodel.pth',
    #                 speaker_recognition_exp_name='notebook_test',
    #                 exp_name='vae-64-lambda-0_5-spkr-recog',
    #                 max_seq_len=1000,
    #                 nspk=22,
    #                 gpu=0,
    #                 batch_size=64,
    #                 seed=1,
    #                 eval_epochs=10,
    #                 b_teacher_force=False,
    #                 b_use_train_noise=False,
    #                 start_epoch=1,
    #                 end_epoch=90,
    #                 step_epoch=1
    #                 )

    # switch back on...
    # !!!!!!!! temporary hack !!!!!!!

    calc_eval_curves(checkpoint_folder='checkpoints/vctk-us-vae-64-mean-lambda-zero-noise-2-final',
                     data='/home/ubuntu/loop/data/vctk',
                     speaker_recognition_checkpoint='/home/ubuntu/msc-project-master/msc-project-master/checkpoints/speaker-recognition-vctk-us/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     exp_name='vae-64-lambda-zero-noise-2-final-spkr-recog',
                     max_seq_len=1000,
                     nspk=22,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     eval_epochs=10,
                     b_teacher_force=False,
                     b_use_train_noise=False,
                     start_epoch=90,
                     end_epoch=180,
                     step_epoch=1
                     )

def old():
    calc_eval_curves(checkpoint_folder='checkpoints/vctk-all',
                     data='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all',
                     speaker_recognition_checkpoint='checkpoints/speaker_recognition_vctk_all/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     exp_name='vctk_all_20180716_teachT_noiseT',
                     max_seq_len=1000,
                     nspk=107,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     eval_epochs=10,
                     b_teacher_force=True,
                     b_use_train_noise=True,
                     start_epoch=1,
                     end_epoch=90,
                     step_epoch=1
                     )

    calc_eval_curves(checkpoint_folder='checkpoints/vctk-all-2-v2',
                     data='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all',
                     speaker_recognition_checkpoint='checkpoints/speaker_recognition_vctk_all/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     exp_name='vctk_all_2_v2_20180716_teachT_noiseT',
                     max_seq_len=1000,
                     nspk=107,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     eval_epochs=10,
                     b_teacher_force=True,
                     b_use_train_noise=True,
                     start_epoch=90,
                     end_epoch=163,
                     step_epoch=1
                     )

    calc_eval_curves(checkpoint_folder='checkpoints/vctk-all',
                     data='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all',
                     speaker_recognition_checkpoint='checkpoints/speaker_recognition_vctk_all/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     exp_name='vctk_all_20180716_teachF_noiseF',
                     max_seq_len=1000,
                     nspk=107,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     eval_epochs=10,
                     b_teacher_force=False,
                     b_use_train_noise=False,
                     start_epoch=1,
                     end_epoch=90,
                     step_epoch=1
                     )

    calc_eval_curves(checkpoint_folder='checkpoints/vctk-all-2-v2',
                     data='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all',
                     speaker_recognition_checkpoint='checkpoints/speaker_recognition_vctk_all/bestmodel.pth',
                     speaker_recognition_exp_name='notebook_test',
                     exp_name='vctk_all_2_v2_20180716_teachF_noiseF',
                     max_seq_len=1000,
                     nspk=107,
                     gpu=0,
                     batch_size=64,
                     seed=1,
                     eval_epochs=10,
                     b_teacher_force=False,
                     b_use_train_noise=False,
                     start_epoch=90,
                     end_epoch=163,
                     step_epoch=1
                     )


    # calc_eval_curves(checkpoint_folder='checkpoints/vctk-us-train-mon',
    #                 data='/home/ubuntu/loop/data/vctk',
    #                 speaker_recognition_checkpoint='checkpoints/speaker-recognition-vctk-us/bestmodel.pth',
    #                 speaker_recognition_exp_name='speaker-recognition-vctk-us',
    #                 max_seq_len=1000,
    #                 nspk=22,
    #                 gpu=0,
    #                 batch_size=64,
    #                 seed=1,
    #                 eval_epochs=2,
    #                 b_teacher_force=False,
    #                 b_use_train_noise=False,
    #                 start_epoch=1,
    #                 end_epoch=10)
