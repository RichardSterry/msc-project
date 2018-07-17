from __future__ import print_function

import os
import pandas as pd

import os
import numpy as np

import matplotlib
import matplotlib.cm as cm

import matplotlib.pyplot as plt


import torch
from model import Loop as Loop_Base

#from model_ident import Loop as Loop_Ident

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

import sklearn.metrics.pairwise as pw

from data import *
from utils import generate_merlin_wav

from torch.autograd import Variable
from IPython.display import Audio
import IPython.display

import phonemizer

from scipy.io import wavfile

import copy

import evaluate_loss_func_for_notebook as el

import IPython.display
from ipywidgets import interact, interactive, fixed

from scipy.io import wavfile

import glob as gl
from IPython.display import display, HTML

import pydub as pyd

import spectrogram as sp

from utils import generate_merlin_wav

from data import *

import shutil as sh

vctk_raw_folder = '/home/ubuntu/VCTK-Corpus/'
vctk_prebuilt_folder = '/home/ubuntu/vctk-16khz-cmu-no-boundaries/'




def get_vctk_speaker_info(vctk_raw_folder='/home/ubuntu/VCTK-Corpus/'):
    # this is the reference data file that comes as part of VCTK
    speaker_info_file = os.path.join(vctk_raw_folder, 'speaker-info.txt')

    # read file contents
    f = open(speaker_info_file, 'r')
    x = f.readlines()
    f.close()

    # extract column headers
    cols = x[0].lower().split()
    num_cols = len(cols)
    del x[0]
    num_speaker = len(x)

    # parse the data line by line
    d = dict()

    for idx in range(num_speaker):
        this_speaker_id = int(x[idx][:3])
        this_age = int(x[idx][5:7])
        this_gender = x[idx][9]
        residual = x[idx][14:].split()
        this_accent = residual[0]
        this_region = " ".join(residual[1:])

        # add speakers to a dictionary
        d[this_speaker_id] = (this_speaker_id, this_age, this_gender, this_accent, this_region)

    # convert to Pandas datafrae
    speaker_info = pd.DataFrame.from_dict(d, orient='index', columns=cols)

    return speaker_info





def plot_spectrogram(wav_data, rate, title="Original Spectrogram"):
    ### Parameters ###
    fft_size = 480  # 2048 # window size for the FFT
    # step_size = int(fft_size/16) # distance to slide along the window (in time)
    step_size = 160  # distance to slide along the window (in time)
    spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 7500  # 15000 # Hz # High cut for our butter bandpass filter

    # For mels
    n_mel_freq_components = 40  # 64 # number of mel frequency channels
    shorten_factor = 1  # 10 # how much should we compress the x-axis (time)
    start_freq = 300  # Hz # What frequency to start sampling our melS from
    end_freq = 8000  # Hz # What frequency to stop sampling our melS from

    data = sp.butter_bandpass_filter(wav_data, lowcut, highcut, rate, order=1)
    # data = butter_bandpass_filter(data, 500, 7500, rate, order=1)
    # Only use a short clip for our demo
    if np.shape(data)[0] / float(rate) > 10:
        data = data[0:rate * 10]
        # print('Length in time (s):' + str(np.shape(data)[0]/float(rate)))

    wav_spectrogram = sp.pretty_spectrogram(wav_data.astype('float64'), fft_size=fft_size,
                                            step_size=step_size, log=True, thresh=spec_thresh)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                     origin='lower')
    fig.colorbar(cax)
    plt.title(title)
    # plt.xlim(0, len(wav_data))
    # plt.xticks(range(0,len(wav_data), 48000), range(0, np.int(np.ceil(1.0*len(wav_data)/48000))))


def plot_spectrogram_comparison(wav_file_a, wav_file_b, label_a='base', label_b='test',
                                title="Original Spectrogram", b_mel=False):
    ### Parameters ###
    fft_size = 480  # 2048 # window size for the FFT
    # step_size = int(fft_size/16) # distance to slide along the window (in time)
    step_size = 160  # distance to slide along the window (in time)
    spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 7500  # 15000 # Hz # High cut for our butter bandpass filter

    # For mels
    n_mel_freq_components = 40  # 64 # number of mel frequency channels
    shorten_factor = 1  # 10 # how much should we compress the x-axis (time)
    start_freq = 300  # Hz # What frequency to start sampling our melS from
    end_freq = 8000  # Hz # What frequency to stop sampling our melS from

    # loop_dict['output_orig_fname']
    # loop_dict['output_fname']

    # load wav file data
    mywav_a = wav_file_a + '.wav'
    mywav_b = wav_file_b + '.wav'

    rate_a, data_a = wavfile.read(mywav_a)
    data_a = sp.butter_bandpass_filter(data_a, lowcut, highcut, rate_a, order=1)

    rate_b, data_b = wavfile.read(mywav_b)
    data_b = sp.butter_bandpass_filter(data_b, lowcut, highcut, rate_b, order=1)

    assert rate_a == rate_b, "Sampling rates don't match"

    # pad
    max_len = np.max([len(data_a), len(data_b)])

    tmp_a = data_a
    data_a = np.zeros(max_len)
    data_a[:len(tmp_a)] = tmp_a

    tmp_b = data_b
    data_b = np.zeros(max_len)
    data_b[:len(tmp_b)] = tmp_b

    # create spectrograms
    wav_spectrogram_a = sp.pretty_spectrogram(data_a.astype('float64'), fft_size=fft_size,
                                              step_size=step_size, log=True, thresh=spec_thresh)

    wav_spectrogram_b = sp.pretty_spectrogram(data_b.astype('float64'), fft_size=fft_size,
                                              step_size=step_size, log=True, thresh=spec_thresh)

    # create mel spectrograms if necessary
    if b_mel:
        # Generate the mel filters
        mel_filter, mel_inversion_filter = sp.create_mel_filter(fft_size=fft_size,
                                                                n_freq_components=n_mel_freq_components,
                                                                start_freq=start_freq,
                                                                end_freq=end_freq,
                                                                samplerate=rate_a)  # may need to change sample rate

        # create mel spectrograms
        mel_spec_a = sp.make_mel(wav_spectrogram_a, mel_filter, shorten_factor=shorten_factor)
        mel_spec_b = sp.make_mel(wav_spectrogram_b, mel_filter, shorten_factor=shorten_factor)

        # use mel spectrograms in the plots (hacky, needs tidying up)
        wav_spectrogram_a = np.float64(mel_spec_a)
        wav_spectrogram_b = np.float64(mel_spec_b)

    # plot spectrograms and the differences
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
    cax = ax.matshow(np.transpose(wav_spectrogram_a), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                     origin='lower')
    f.colorbar(cax)
    plt.title('Spectrogram: ' + wav_file_a + '(' + label_a + ')')

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
    cax = ax.matshow(np.transpose(wav_spectrogram_b), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                     origin='lower')
    f.colorbar(cax)
    plt.title('Spectrogram: ' + wav_file_b + '(' + label_b + ')')

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
    cax = ax.matshow(np.transpose(wav_spectrogram_b - wav_spectrogram_a), interpolation='nearest', aspect='auto',
                     cmap=plt.cm.afmhot, origin='lower')
    f.colorbar(cax)
    plt.title('Difference: (' + wav_file_b + ' - ' + wav_file_a + ') (' + label_b + ' - ' + label_a + ')')


def plot_waveform_comparison(wav_file_a, wav_file_b, label_a='base', label_b='test',
                             title="Original Spectrogram", b_mel=False):
    # load wav file data
    mywav_a = wav_file_a + '.wav'
    mywav_b = wav_file_b + '.wav'

    rate_a, data_a = wavfile.read(mywav_a)

    rate_b, data_b = wavfile.read(mywav_b)

    assert rate_a == rate_b, "Sampling rates don't match"

    # pad
    max_len = np.max([len(data_a), len(data_b)])

    tmp_a = data_a
    data_a = np.zeros(max_len)
    data_a[:len(tmp_a)] = tmp_a

    tmp_b = data_b
    data_b = np.zeros(max_len)
    data_b[:len(tmp_b)] = tmp_b

    max_a = data_a.max()
    min_a = data_a.min()
    max_b = data_b.max()
    min_b = data_b.min()

    max_both = max([max_a, max_b])
    min_both = min([min_a, min_b])

    plt.figure(figsize=(15, 6))

    ax = plt.subplot(3, 1, 1)
    plt.plot(data_a)
    plt.grid(True)
    plt.xlim(0, len(data_a))
    plt.ylim(min_both, max_both)
    plt.gca().set_xticks(range(0, len(data_a), rate_a), range(0, np.int(np.ceil(1.0 * len(data_a) / rate_a))))
    plt.title(label_a)
    plt.ylabel('Amplitude')

    ax = plt.subplot(3, 1, 2)
    plt.plot(data_b)
    plt.grid(True)
    plt.xlim(0, len(data_b))
    plt.ylim(min_both, max_both)
    ax.set_xticks(range(0, len(data_b), rate_b), range(0, np.int(np.ceil(1.0 * len(data_b) / rate_b))))
    plt.title(label_b)
    plt.ylabel('Amplitude')

    ax = plt.subplot(3, 1, 3)
    plt.plot(data_b - data_a)
    plt.grid(True)
    plt.xlim(0, len(data_b))
    plt.ylim(min_both, max_both)
    ax.set_xticks(range(0, len(data_b), rate_b), range(0, np.int(np.ceil(1.0 * len(data_b) / rate_b))))
    plt.title('Diff: ' + label_b + ' - ' + label_a)
    plt.ylabel('Amplitude')


def plot_mgc_feature_comparison(feats_a, feats_b, label_a, label_b):
    # plot mgc features
    plt.figure(figsize=(20, 20))
    for x in range(60):
        plt.subplot(10, 6, x + 1)
        plt.plot(feats_a[:, x])
        plt.plot(feats_b[:, x])
        plt.title('mgc' + str(x))
        if x == 5:
            plt.legend([label_a, label_b])
        if x < 6 * 9:
            plt.gca().get_xaxis().set_visible(False)
        if not np.mod(x, 6) == 0:
            plt.gca().get_yaxis().set_visible(False)

    plt.suptitle('mgc features', fontsize=14)
    plt.show()


def plot_other_feature_comparison(feats_a, feats_b, label_a, label_b):
    # plot lf0, vuv and bap features
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 3, 1)
    plt.plot(feats_a[:, 60])
    plt.plot(feats_b[:, 60])
    plt.title('vuv')
    plt.grid(True)
    plt.legend([label_a, label_b])

    plt.subplot(1, 3, 2)
    plt.plot(feats_a[:, 61])
    plt.plot(feats_b[:, 61])
    plt.title('lf0')
    plt.grid(True)
    plt.legend([label_a, label_b])

    plt.subplot(1, 3, 3)
    plt.plot(feats_a[:, 62])
    plt.plot(feats_b[:, 62])
    plt.title('bap')
    plt.grid(True)
    plt.legend([label_a, label_b])

    plt.show()


def load_pre_calc_features(speaker_id, sample_id, b_valid=False, b_original=True):
    if b_original:
        pre_calc_features_folder = '/home/ubuntu/loop/data/vctk/numpy_features/'
    else:
        pre_calc_features_folder = '/home/ubuntu/vctk-16khz-cmu-no-boundaries/numpy_features/'

    if b_valid:
        pre_calc_features_folder = pre_calc_features_folder.replace('_features', '_features_numpy')

    pre_calc_features_file = os.path.join(pre_calc_features_folder,
                                          'p' + str(speaker_id) + '_' + '{num:03d}'.format(num=sample_id) + '.npz')

    feats = np.load(pre_calc_features_file)

    return feats


def play_synthesized_features(feats, norm_path='/home/ubuntu/loop/data/vctk/norm_info/norm.dat'):
    output_dir = './'
    output_file = 'test.wav'

    generate_merlin_wav(feats['audio_features'],
                        output_dir,
                        output_file,
                        norm_path)

    IPython.display.display(IPython.display.Audio(output_file + '.wav', autoplay=True))

    rate, wav_data = wavfile.read(output_file + '.wav')

    return rate, wav_data


def display_vctk_sample(vctk_speaker_id, sample_id):
    speaker_info = get_vctk_speaker_info()

    # raw .wav file
    wav_file = os.path.join(vctk_raw_folder,
                            'wav48/p' + str(vctk_speaker_id) + '/p' + str(vctk_speaker_id) + '_' + '{num:03d}'.format(
                                num=sample_id) + '.wav')

    # print speaker info
    display(speaker_info[speaker_info['id'] == vctk_speaker_id])

    # print text
    txt_file = wav_file.replace("wav48", "txt").replace(".wav", ".txt")
    f = open(txt_file, 'r')
    print(f.read())
    f.close()

    # play sample
    rate, wav_data = wavfile.read(wav_file)
    display(IPython.display.Audio(data=wav_data, rate=rate, autoplay=True))

    # plot waveform
    plt.figure(figsize=(15, 4))
    plt.plot(wav_data)
    plt.grid(True)
    plt.xlim(0, len(wav_data))
    plt.xticks(range(0, len(wav_data), 48000), range(0, np.int(np.ceil(1.0 * len(wav_data) / 48000))))

    # plot spectrogram
    plot_spectrogram(wav_data, rate)



def generate_sample_with_loop(npz='', text='', spkr_id=1, gender=1,
                              checkpoint='models/vctk-16khz-cmu-no-boundaries-all/bestmodel.pth',
                              output_dir='./',
                              npz_path='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all/numpy_features',
                              output_file_override=None,
                              ident_override=None):
    # npz = ''
    # text = 'Your tickets for the social issues'
    # text = 'see that girl watch that scene'
    # npz = '/home/ubuntu/loop/data/vctk/numpy_features/p294_011.npz'
    # spkr_id = 12
    # checkpoint = 'checkpoints/vctk/lastmodel.pth'
    # checkpoint = 'models/vctk/bestmodel.pth'


    gender = np.array(gender).reshape(-1)
    out_dict = dict()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpu = 0

    # load loop weights & params from checkpoint
    weights = torch.load(checkpoint,
                         map_location=lambda storage, loc: storage)
    opt = torch.load(os.path.dirname(checkpoint) + '/args.pth')
    train_args = opt[0]

    train_dataset = NpzFolder('/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all/numpy_features')
    char2code = train_dataset.dict
    spkr2code = train_dataset.speakers
    # print spkr2code.cpu().data

    norm_path = train_args.data + '/norm_info/norm.dat'
    norm_path = '/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all/norm_info/norm.dat'
    train_args.noise = 0

    valid_dataset_path = npz_path + '_valid'

    # prepare loop model
    if ident_override:
        model = Loop_Ident(train_args)
    else:
        model = Loop_Base(train_args)

    model.load_state_dict(weights)
    if gpu >= 0:
        model.cuda()
    model.eval()

    # check speaker id is valid
    if spkr_id not in range(len(spkr2code)):
        print('ERROR: Unknown speaker id: %d.' % spkr_id)

    # get phone sequence
    txt, feat, spkr, output_fname = None, None, None, None
    if npz is not '':
        # use pre-calculated phonemes etc.
        txt, feat, pre_calc_feat = npy_loader_phonemes(os.path.join(npz_path, npz))

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(torch.LongTensor([spkr_id]), volatile=True)

        output_file = os.path.basename(npz)[:-4] + '_' + str(spkr_id)

        out_dict['pre_calc_feat'] = pre_calc_feat

    elif text is not '':
        # use specified text string
        # extract phonemes from the text
        txt = text2phone(text, char2code)
        feat = torch.FloatTensor(txt.size(0) * 20, 63)
        spkr = torch.LongTensor([spkr_id])

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(spkr, volatile=True)

        output_file = text.replace(' ', '_')
    else:
        print('ERROR: Must supply npz file path or text as source.')
        raise Exception('Need source')

    if output_file_override:
        output_file = output_file_override

    # use gpu
    if gpu >= 0:
        txt = txt.cuda()
        feat = feat.cuda()
        spkr = spkr.cuda()

    # run loop model to generate output features
    # print(ident_override)
    if ident_override:
        loop_feat, attn = model([txt, spkr, gender], feat, ident_override=ident_override)
    else:
        loop_feat, attn = model([txt, spkr, gender], feat)

    loop_feat, attn = trim_pred(loop_feat, attn)

    # add to output dictionary
    out_dict['txt'] = txt[:, 0].squeeze().data.tolist()
    out_dict['spkr'] = spkr
    out_dict['feat'] = feat.data.cpu().numpy()
    out_dict['loop_feat'] = loop_feat.data.cpu().numpy()
    out_dict['attn'] = attn.squeeze().data.cpu().numpy()
    out_dict['output_file'] = output_file
    out_dict['valid_dataset_path'] = valid_dataset_path

    # print output_dir

    # generate .wav file from loop output features
    #print(output_dir)
    #print(output_file)
    #print(norm_path)

    generate_merlin_wav(loop_feat.data.cpu().numpy(),
                        output_dir,
                        output_file,
                        norm_path)

    # generate .wav file from original features for reference
    if npz is not '':
        output_orig_fname = os.path.basename(npz)[:-4] + '.orig'
        generate_merlin_wav(feat[:, 0, :].data.cpu().numpy(),
                            output_dir,
                            output_orig_fname,
                            norm_path)
        out_dict['output_orig_fname'] = output_orig_fname

    return out_dict


def generate_samples_for_spkr_list(spkr_id_list,
                                   npz='',
                                   text='',
                                   checkpoint='checkpoints/vctk-16khz-cmu-no-boundaries-all-noise-2/bestmodel.pth',
                                   output_dir='./',
                                   npz_path='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all/numpy_features'):
    out = []
    for spkr in spkr_id_list:
        output_file_override = 'gen_test_' + str(spkr)
        loop_dict = generate_sample_with_loop(spkr_id=spkr,
                                              npz=npz,
                                              text=text,
                                              checkpoint=checkpoint,
                                              output_dir='./',
                                              npz_path='/home/ubuntu/loop/data/vctk-16khz-cmu-no-boundaries-all/numpy_features',
                                              output_file_override=output_file_override)

        out.append(loop_dict)
        IPython.display.display(Audio(loop_dict['output_file'] + '.wav', autoplay=True))

    return out


def text2phone(text, char2code):
    seperator = phonemizer.separator.Separator('', '', ' ')
    ph = phonemizer.phonemize(text, separator=seperator)
    ph = ph.split(' ')
    ph.remove('')

    ph = [p.replace('zh', 'jh') for p in ph]

    result = [char2code[p] for p in ph]
    return torch.LongTensor(result)


def trim_pred(out, attn):
    tq = attn.abs().sum(1).data

    for stopi in range(1, tq.size(0)):
        col_sum = attn[:stopi, :].abs().sum(0).data.squeeze()

        if type(tq[stopi]) == float:
            if tq[stopi] < 0.5 and col_sum[-1] > 4:
                break
        else:
            if tq[stopi][0] < 0.5 and col_sum[-1] > 4:
                break

    out = out[:stopi, :]
    attn = attn[:stopi, :]

    return out, attn


def npy_loader_phonemes(path):
    feat = np.load(path)

    txt = feat['phonemes'].astype('int64')
    txt = torch.from_numpy(txt)

    audio = feat['audio_features']
    audio = torch.from_numpy(audio)

    return txt, audio, feat


def plot_attn(data, labels, dict_file):
    labels_dict = dict_file
    labels_dict = {v: k for k, v in labels_dict.iteritems()}
    labels = [labels_dict[x].decode('latin-1') for x in labels]

    plt.figure(figsize=(15, 15))
    axarr = plt.subplot()
    axarr.imshow(data.T, aspect='auto', origin='lower', interpolation='nearest', cmap=cm.viridis)
    axarr.set_yticks(np.arange(0, len(data.T)))
    axarr.set_yticklabels(labels, rotation=90)


def plot_pca_by_accent(pca_fit, speaker_info_loop, u_accents_to_show=None, gender_to_show='all'):
    idx_m = speaker_info_loop['gender'] == 'M'
    idx_f = speaker_info_loop['gender'] == 'F'

    title_str = 'PCA: Speakers by Accent'

    if u_accents_to_show is None:
        u_accents_to_show = speaker_info_loop.accents.unique()

    if gender_to_show == 'm':
        idx_gender_to_show = idx_m
        title_str += ' (male)'
    elif gender_to_show == 'f':
        idx_gender_to_show = idx_f
        title_str += ' (female)'
    else:
        idx_gender_to_show = idx_m == idx_m

    for u in u_accents_to_show:
        idx_this_accent = (speaker_info_loop.accents == u) & idx_gender_to_show
        # print(str(idx_this_accent.sum()))
        # print str(np.sum(idx_this_accent))

        plt.scatter(pca_fit[idx_this_accent, 0], pca_fit[idx_this_accent, 1])
        # f_plot = plt.scatter(S_tsne[idx_other, 0], S_tsne[idx_other, 1], c='r', label='other')

    plt.legend(u_accents_to_show, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.title(title_str)
    plt.show()
