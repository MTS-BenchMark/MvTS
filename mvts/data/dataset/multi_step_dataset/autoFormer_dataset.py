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

from mvts.data.dataset.multi_step_dataset import MultiStepDataset
from mvts.data.utils import time_features, DataLoader_transformer


class AutoFormerDataset(MultiStepDataset):

    def __init__(self, config):
        
        self.config = config

        self.timeList_gene = self.config.get("timeList_gene")
        self.seq_len = self.config.get("seq_len", 0)
        self.embed = self.config.get("embed")
        self.freq = self.config.get("freq")
        self.label_len = self.config.get("label_len")
        
        timeenc = 0 if self.embed != 'timeF' else 1
        self.timeenc = timeenc

        super().__init__(config)
    

    def _get_timeList(self):


        res = {"date": self.time}

        df_stamp = pd.DataFrame(data=res)

        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        return data_stamp
        

    def _gene_dataset(self):
        data = {}
       
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

        self.scaler = self._get_scalar(x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_valid[..., :self.output_dim] = self.scaler.transform(x_valid[..., :self.output_dim])
        y_valid[..., :self.output_dim] = self.scaler.transform(y_valid[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])


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
            "num_batches": self.data['num_batches']
        }

        return feature









