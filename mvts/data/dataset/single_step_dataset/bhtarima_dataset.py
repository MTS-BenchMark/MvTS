import torch
import os
import h5py
import numpy as np
from mvts.data.dataset import AbstractDataset


class BHTARIMADataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, config):
        self.config = config
        self.filename = self.config.get("filename", "")


    def get_data(self):
        # ori_ts = np.load(self.filename)['data'].T  # [items, times]
        f = h5py.File(self.filename, "r")
        ori_ts = np.array(f["raw_data"])
        print(ori_ts.shape)
        seq_len, num_nodes, feature_dim = ori_ts.shape
        ori_ts = ori_ts.reshape(seq_len, num_nodes * feature_dim).T #[items, times]
        self.ts = ori_ts[..., :-1]  # training data, [items, times-1]
        self.label = ori_ts[..., -1]
        return [self.ts, self.label], None, None


    def get_data_feature(self):
        self.ts_ori_shape = self.ts.shape
        self.N = len(self.ts.shape) - 1  # 1
        self.T = self.ts.shape[-1]
        feature = {"ts_ori_shape": self.ts_ori_shape,
                   "N": self.N,
                   "T": self.T}
        return feature