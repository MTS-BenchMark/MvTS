import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import h5py

from mvts.data.dataset import AbstractDataset
from mvts.data.utils import DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


class SingleStepDataset(AbstractDataset):

    def __init__(self, config):

        self.config = config
        file_name = self.config.get("filename", " ")
        cuda = self.config.get("cuda", True)
        horizon = self.config.get("horizon", 12)
        window = self.config.get("window", 168)
        normalize = self.config.get("normalize", 2)
        self.adj_type = self.config.get("adj_type", "distance")
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
            self.adj_mx = None
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
            self.adj_mx = None
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


    def _gene_dataset(self):
        train = self.config.get("train_rate", 0.6)
        valid = self.config.get("eval_rate", 0.2)

        train_dat = self.dat[:round(train * self.n), ...]
        scaler = self._get_scalar(train_dat, train_dat)
        self.dat = scaler.transform(self.dat)
        
        train, valid, test = self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        data = {}
        data["x_train"], data["y_train"] = train[0], train[1]
        data["x_valid"], data["y_valid"] = valid[0], valid[1]
        data["x_test"], data["y_test"] = test[0], test[1]
        
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
        feature = {"scaler": self.data["scaler"], "time": self.time, "adj_mx": self.adj_mx}

        return feature
