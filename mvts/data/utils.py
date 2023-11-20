import importlib
import numpy as np
import copy
import pickle
import torch.utils.data as torch_data
import torch
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List


from mvts.data.list_dataset import ListDataset

def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    if config["task"] == "multi_step":
        try:
            return getattr(importlib.import_module('mvts.data.dataset.multi_step_dataset'),
                        config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')
    elif config["task"] == "single_step":
        try:
            return getattr(importlib.import_module('mvts.data.dataset.single_step_dataset'),
                config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.seq_len = ys.shape[0]
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class DataLoaderM_new(object):
    def __init__(self, xs, ys, ycl, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ycl = np.concatenate([ycl, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ycl = ycl

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, ycl = self.xs[permutation], self.ys[permutation], self.ycl[
            permutation]
        self.xs = xs
        self.ys = ys
        self.ycl = ycl

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                y_i_cl = self.ycl[start_ind:end_ind, ...]
                yield (x_i, y_i, y_i_cl)
                self.current_ind += 1

        return _wrapper()
        

class DataLoader_transformer(object):
    def __init__(self, xs, ys, x_mark, y_mark, batch_size, pad_with_last_sample=False, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        assert len(xs) == len(x_mark)
        self.batch_size = batch_size
        self.current_ind = 0
        self.seq_len = ys.shape[0]
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x_mark_padding = np.repeat(x_mark[-1:], num_padding, axis=0)
            y_mark_padding = np.repeat(y_mark[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            xs = np.concatenate([x_mark, x_mark_padding], axis=0)
            ys = np.concatenate([y_mark, y_mark_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
            x_mark, y_mark = x_mark[permutation], y_mark[permutation]
        self.xs = xs
        self.ys = ys
        self.x_mark = x_mark
        self.y_mark = y_mark


    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        x_mark, y_mark = self.x_mark[permutation], self.y_mark[permutation]
        self.xs = xs
        self.ys = ys
        self.x_mark = x_mark
        self.y_mark = y_mark

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_mark_i = self.x_mark[start_ind: end_ind, ...]
                y_mark_i = self.y_mark[start_ind: end_ind, ...]
                yield (x_i, y_i, x_mark_i, y_mark_i)
                self.current_ind += 1

        return _wrapper()



