import os
import sys
import numpy as np
import pandas as pd
from IPython.display import display
import datetime as dt
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

def datetime_to_float(d):
    epoch = dt.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds


class TrainingMonitor(object):
    def __init__(self, file, exp_name, b_append=True, path='training_logs',
                 columns=('epoch', 'update_time', 'train_loss', 'valid_loss', 'mcd'),
                 source_file=None):
        self.path = path
        self.file = os.path.join(path, os.path.splitext(file)[0]) + ".csv"
        self.exp_name = exp_name
        self.columns = columns
        self.data = pd.DataFrame(columns=self.columns)
        self.data.exp_name = exp_name
        self.source_file = source_file
        self.source_file_by_row = None

        if b_append:
            if os.path.isfile(self.file):
                self.read()

    #def __repr__(self):
        # wrong...
    #    self.disp()
    #    return ""

    def read(self):
        if self.source_file:
            #self.data = pd.read_csv(self.source_file[0])
            #for i in range(1, len(self.source_file)):
            #    self.append(pd.read_csv(self.source_file[i]))
            self.source_file_by_row = []
            for f in self.source_file:
                new_data = pd.read_csv(os.path.join(self.path, f))
                self.append(new_data)
                [self.source_file_by_row.append(l) for l in np.repeat(f, len(new_data)).tolist()]
        else:
            self.data = pd.read_csv(self.file)

    def write(self):
        if not os.path.isdir(os.path.dirname(self.file)):
            os.mkdir(os.path.dirname(self.file))
        self.data.to_csv(self.file, encoding='utf-8', index=False)

    def append(self, new):
        # TODO: check new; potentially add only some columns; check key
        if isinstance(new, pd.core.frame.DataFrame):
            new_df = new
        else:
            new_df = pd.DataFrame(new, columns=self.columns)
        #self.data = self.data.append(new_df)
        self.data = pd.concat([self.data, new_df], axis=0, sort=True).reset_index(drop=True)


    def insert(self, epoch, train_loss=None, valid_loss=None, mcd=None, train_acc=None, valid_acc=None, speaker_recognition_acc_eval=None,
               update_time=None):
        if not update_time:
            update_time = datetime_to_float(dt.datetime.now())

        new_df = pd.DataFrame(np.hstack([vars()[x] for x in self.columns]).reshape(1,-1), columns=self.columns)
        #self.data = self.data.append(new_df)
        #self.data = new_df.combine_first(self.data)
        #self.data.update(new_df, join='epoch')
        idx_kill = self.data.epoch == epoch
        self.data = self.data[~idx_kill]
        self.data = self.data.append(new_df)
        #self.data.sort_index(axis=1, inplace=True)
        self.data = self.data.sort_values(by='epoch').reset_index(drop=True)


    def disp(self, b_show_all=False):
        tmp = self.data.copy()
        tmp.update_time = [dt.datetime.fromtimestamp(x).strftime("%d-%b-%Y %H:%M:%S") for x in tmp.update_time]
        if b_show_all:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(tmp)
        else:
            display(tmp)

    def plot_loss(self, ax=None):
        if ax:
            plt.sca(ax)
            title_str = 'Training Loss Curve'
        else:
            title_str = 'Training Loss Curve for ' + self.exp_name

        plt.plot(self.data.epoch, self.data.train_loss.fillna(method='ffill'))
        plt.plot(self.data.epoch, self.data.valid_loss.fillna(method='ffill'))
        plt.grid(True)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train', 'Validation'))
        plt.title(title_str)
        if not ax:
            plt.show()

    def plot_mcd(self, ax=None):
        if ax:
            plt.sca(ax)
            title_str = 'Training MCD Curve'
        else:
            title_str = 'Training MCD Curve for ' + self.exp_name

        plt.plot(self.data.epoch, self.data.mcd.fillna(method='ffill'))
        plt.grid(True)
        plt.ylabel('MCD')
        plt.xlabel('Epoch')
        plt.title(title_str)
        if not ax:
            plt.show()

    def plot_speaker_recognition(self, ax=None):
        if ax:
            plt.sca(ax)
            title_str = 'Training Speaker Recognition Accuracy Curve'
        else:
            title_str = 'Training Speaker Recognition Accuracy Curve for ' + self.exp_name

        plt.plot(self.data.epoch, self.data.speaker_recognition_acc_eval.fillna(method='ffill'))
        plt.grid(True)
        plt.ylabel('Speaker Recognition Accuracy')
        plt.xlabel('Epoch')
        plt.title(title_str)
        if not ax:
            plt.show()

    def plot(self, figsize=(16,6)):
        f, ax = plt.subplots(1, 3, figsize=figsize)
        self.plot_loss(ax[0])
        self.plot_mcd(ax[1])
        self.plot_speaker_recognition(ax[2])
        plt.suptitle(self.exp_name)
        plt.show()

    def combine(self, other, new_exp_name):
        tm = TrainingMonitor(file=new_exp_name, exp_name=new_exp_name, b_append=False, path=os.path.dirname(self.file), columns=self.columns, source_file=self.source_file)
        tm.data = self.data.copy()
        tm.append(other.data)
        return tm


class LossDecomposition(object):
    def __init__(self, training_monitor):
        #assert isinstance(training_monitor, TrainingMonitor), "LossDecomposition takes a TrainingMonitor as arg"
        self.training_monitor = training_monitor
        self.all_loss_contribs = []
        self.all_loss_epochs = []
        self.loss_by_feature = None

    def get_loss_contrib(self, epoch=None):
        max_epoch = self.training_monitor.data.epoch.max()
        if not epoch:
            epoch = max_epoch

        loss_file = os.path.join('training_logs', self.training_monitor.source_file_by_row[epoch][:-4] + '_loss_contrib_' + str(epoch) + '.pickle')
        with open(loss_file, 'rb') as handle:
            loss_contrib = pickle.load(handle)

        return loss_contrib

    def load_all_loss_contribs(self, epoch_list=None):
        self.all_loss_contribs = []
        self.all_loss_epochs = []
        if not epoch_list:
            epoch_list = self.training_monitor.data.epoch.astype(np.int)

        for epoch in epoch_list:
            #try:
                tmp = self.get_loss_contrib(epoch)
                self.all_loss_contribs.append(tmp)
                self.all_loss_epochs.append(epoch)
            #except:
            #    print type(epoch)
            #    pass

    def get_loss_by_feature(self, epoch_list=None):
        #if not self.all_loss_contribs:
        #    self.load_all_loss_contribs()

        if not epoch_list:
            #epoch_list = self.training_monitor.data.epoch.astype(np.int)
            idx = np.where(np.logical_not(np.isnan(self.training_monitor.data.mcd)))
            epoch = self.training_monitor.data.iloc[idx].epoch
            epoch_list = epoch.astype(int).tolist()

        num_epoch = len(epoch_list)
        self.loss_by_feature = np.zeros((num_epoch, 63))
        self.epoch_list = epoch_list

        for i, epoch in enumerate(epoch_list):
            try:
                tmp = self.get_loss_contrib(epoch)
                self.loss_by_feature[i, :] = tmp.sum((0, 1))
                print "Done " + str(epoch)
            except:
                print "Failed for " + str(epoch)

        return self.loss_by_feature

    def plot_loss_by_feature(self, figsize=(14,8)):
        if self.loss_by_feature is None:
            self.get_loss_by_feature()

        plt.figure(figsize=figsize)
        plt.plot(self.loss_by_feature.transpose(), linestyle="None", marker="+")
        plt.grid(True)
        plt.legend(['epoch %d' % e for e in self.epoch_list])
        plt.title('Loss by Feature: ' + self.training_monitor.exp_name)
        plt.ylabel('MSE Loss')
        plt.xlabel('WORLD Feature')
        plt.show()

