import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List
import h5py

from mvts.data.dataset import AbstractDataset
from mvts.data.utils import DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir
from mvts.data.utils import time_features, DataLoader_transformer


class AutoFormerDataset(AbstractDataset):

    def __init__(self, config):
        
        self.config = config

        self.file_name = self.config.get("filename", " ")
        self.adj_filename = self.config.get("adj_filename", "")
        self.adj_type = self.config.get("adj_type", None)

        self.train_rate = self.config.get("train_rate", 0.6)
        self.valid_rate = self.config.get("eval_rate", 0.2)
        self.cuda = self.config.get("cuda", True)

        self.horizon = self.config.get("horizon", 3)
        self.window = self.config.get("window", 12)

        self.normalize = self.config.get("normalize", 2)
        self.batch_size = self.config.get("batch_size", 64)
        self.adj_mx = None
        self.add_time_in_day = self.config.get("add_time_in_day", False)
        self.add_day_in_week = self.config.get("add_day_in_week", False)
        self.input_dim = self.config.get("input_dim", 1)
        self.output_dim = self.config.get("output_dim", 1)
        #self.ensure_adj_mat()
        self._load_origin_data(self.file_name, self.adj_filename)

        self.timeList_gene = self.config.get("timeList_gene")
        self.seq_len = self.config.get("seq_len", 0)
        self.embed = self.config.get("embed")
        self.freq = self.config.get("freq")
        self.label_len = self.config.get("label_len")
        
        timeenc = 0 if self.embed != 'timeF' else 1
        self.timeenc = timeenc

        self.data = self._gene_dataset()
    

    def _load_origin_data(self, file_name, adj_name):
        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
        elif file_name[-2:] == "h5":
            # self.rawdat = pd.read_hdf(file_name)
            f = h5py.File(file_name, "r")
            data = np.array(f["raw_data"])
            adj = np.array(f["adjacency_matrix"])
            
            time = np.array(f["time"])
            t = []
            for i in range(time.shape[0]):
                t.append(time[i].decode())
            time = np.stack(t, axis=0)
            time = pd.to_datetime(time)

            self.rawdat = data
            self.time = time

            if self.adj_type == "distance":
                self.adj_mx = adj
            else:
                row, col = adj.shape
                for i in range(row):
                    for j in range(i, col):
                        if adj[i][j] > 0:
                            adj[i][j] = 1
                            adj[j][i] = 1
                        else:
                            adj[i][j] = 0
                            adj[j][i] = 0
                self.adj_mx = adj
        elif file_name[-3:] == "npz":
            mid_dat = np.load(file_name)
            self.rawdat = mid_dat[mid_dat.files[0]]
        else:
            raise ValueError('file_name type error!')

        self.rawdat = self.rawdat.reshape([self.rawdat.shape[0], -1])


    def _get_scalar(self, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if self.normalize == 2:
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            print('NormalScaler max: ' + str(scaler.max))
        elif self.normalize == 1:
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            print('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif self.normalize == 3:
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            print('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.normalize == 4:
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            print('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.normalize == 5:
            scaler = LogScaler()
            print('LogScaler')
        elif self.normalize == 6:
            data_min = np.min(x_train, axis=0).tolist()
            data_max = np.max(x_train, axis=0).tolist()
            scaler = MinMax01Scaler(maxx=data_max, minn=data_min, column_wise=True)
            # print('List MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.normalize == 7:
            mean = np.mean(x_train, axis=0).tolist()
            std = np.std(x_train, axis=0).tolist()
            std = [1 if i == 0 else i for i in std]
            scaler = StandardScaler(mean=mean, std=std, column_wise=True)
            # print('List StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif self.normalize == 8:
            maxlist = np.max(np.abs(x_train), axis=0).tolist()
            scaler = NormalScaler(maxx=maxlist, column_wise=True)
        elif self.normalize == 0:
            scaler = NoneScaler()
            print('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler


    def _get_timeList(self):

        time = []
        for i in range(self.seq_len):
            time.append(i * self.timeList_gene["time_step"])
        
        res = pd.to_datetime(
            time,
            unit=self.timeList_gene["unit"],
            origin=self.timeList_gene["origin"]
        )

        res = {"date": res}

        df_stamp = pd.DataFrame(data=res)

        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date']   .values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        return data_stamp
        

    def _generate_graph_seq2seq_io_data(
            self, df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
    ):

        num_samples, num_nodes = df.shape[0], df.shape[1]
        if not isinstance(df, np.ndarray):
            data = np.expand_dims(df.values, axis=-1)
            data_list = [data]
        else:
            data_list = [df]
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)

        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)

        return x, y

    def _generate_train_val_test(self, data= None, add_time_in_day=None, add_day_in_week=None):
        if data is None:
            data = self.rawdat
        if add_day_in_week is None:
            add_day_in_week = self.add_day_in_week
            add_time_in_day = self.add_time_in_day
        seq_length_x, seq_length_y = self.window, self.horizon
        x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
        y_offsets = np.arange(seq_length_y, (seq_length_y + 1), 1)
        x, y = self._generate_graph_seq2seq_io_data(data, x_offsets,
                                                    y_offsets, add_time_in_day, add_day_in_week)
        print("x shape: ", x.shape, ", y shape: ", y.shape)
        num_samples = x.shape[0]
        num_val = round(num_samples * self.valid_rate)
        num_train = round(num_samples * self.train_rate)
        num_test = num_samples - num_train - num_val
        # num_val = 672
        # num_test = 672
        # num_train = num_samples - num_val - num_test
        return [x[:num_train], y[:num_train]], \
               [x[num_train:num_train + num_val], y[num_train:num_train + num_val]], \
               [x[num_train + num_val:], y[num_train + num_val:]]


    def _gene_dataset(self):
        data = {}

        _train = self.config.get("train_rate", 0.6)
        train_dat = self.rawdat[:int(_train * self.seq_len), ...]
        self.scaler = self._get_scalar(train_dat, train_dat)
        self.rawdat = self.scaler.transform(self.rawdat)
        self.rawdat = np.expand_dims(self.rawdat, -1)
       
        self.train, self.valid, self.test = self._generate_train_val_test(self.rawdat)
        self.train_mark, self.valid_mark, self.test_mark = self._generate_train_val_test(self._get_timeList(), False, False)

        x_train, y_train = self.train[0], np.concatenate((self.train[0][:, -self.label_len:, ...], self.train[1]), axis=1)
        x_valid, y_valid = self.valid[0], np.concatenate((self.valid[0][:, -self.label_len:, ...], self.valid[1]), axis=1)
        x_test, y_test = self.test[0], np.concatenate((self.test[0][:, -self.label_len:, ...], self.test[1]), axis=1)

        x_mark_train, y_mark_train = self.train_mark[0], np.concatenate(
            (self.train_mark[0][:, -self.label_len:, ...], self.train_mark[1]), axis=1)
        x_mark_valid, y_mark_valid = self.valid_mark[0], np.concatenate(
            (self.valid_mark[0][:, -self.label_len:, ...], self.valid_mark[1]), axis=1)
        x_mark_test, y_mark_test = self.test_mark[0], np.concatenate(
            (self.test_mark[0][:, -self.label_len:, ...], self.test_mark[1]), axis=1)


        data['train_loader'] = DataLoader_transformer(x_train[..., :self.input_dim], y_train[..., :self.output_dim],
                                                x_mark_train, y_mark_train, self.batch_size)
        data['valid_loader'] = DataLoader_transformer(x_valid[..., :self.input_dim], y_valid[..., :self.output_dim],
                                                x_mark_valid, y_mark_valid, self.batch_size)
        data['test_loader'] = DataLoader_transformer(x_test[..., :self.input_dim], y_test[..., :self.output_dim], 
                                                x_mark_test, y_mark_test, self.batch_size)
        data['scaler'] = self.scaler
        data['num_batches'] = x_train.shape[0] / self.batch_size
        return data


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {
            "scaler": self.data["scaler"],
            "adj_mx": self.adj_mx,
            "num_batches": self.data['num_batches'],
            "time": self.time
        }

        return feature









