import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import DataLoader
from libcity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


class SingleStepDataset(AbstractDataset):

    def __init__(self, config):

        self.config = config
        file_name = self.config.get("filename", " ")
        cuda = self.config.get("cuda", True)
        horizon = self.config.get("horizon", 12)
        window = self.config.get("window", 168)
        normalize = self.config.get("normalize", 2)
        self._load_origin_data(file_name)
        self.cuda = cuda
        self.P = window
        self.h = horizon
        self.dat = self.rawdat
        self.n, self.m = self.dat.shape  # seq, node
        self.normalize = normalize
        self.batch_size = self.config.get("batch_size", 64)

        self.data = self._gene_dataset()


    def _load_origin_data(self, file_name):
        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
        elif file_name[-2:] == "h5":
            self.rawdat = pd.read_hdf(file_name).values
        else:
            raise ValueError('file_name type error!')


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
        elif self.normalize == 0:
            scaler = NoneScaler()
            print('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def _gene_dataset(self):
        train = self.config.get("train_rate", 0.6)
        valid = self.config.get("eval_rate", 0.2)
        
        train, valid, test = self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        data = {}
        data["x_train"], data["y_train"] = train[0], train[1]
        data["x_valid"], data["y_valid"] = valid[0], valid[1]
        data["x_test"], data["y_test"] = test[0], test[1]
        scaler = self._get_scalar(data["x_train"], data["y_train"])
        for category in ['train', 'valid', 'test']:
            data['x_' + category] = scaler.transform(data['x_' + category])
            data['y_' + category] = scaler.transform(data['y_' + category])
        
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], self.batch_size)
        data['valid_loader'] = DataLoader(data['x_valid'], data['y_valid'], self.batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], self.batch_size)
        data['scaler'] = scaler
        del train, valid, test
        return data
    

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        return self._batchify(train_set, self.h), self._batchify(valid_set, self.h), self._batchify(test_set, self.h)
    

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = []
        Y = []

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X.append(self.dat[start:end, ...])
            Y.append(self.dat[idx_set[i], ...])
        
        X = np.stack(X, axis=0)  # [B, T, N ,C]
        Y = np.stack(Y, axis=0)  # [B, N, C]

        return [X, Y]




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
        feature = {"scaler": self.data["scaler"]}

        return feature
