import torch
import h5py
import pandas as pd
import numpy as np
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class MTGNNDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        file_name = self.config.get("filename", " ")
        train = self.config.get("train_rate", 0.6)
        eval = self.config.get("eval_rate", 0.2)
        valid = 1 - train - eval
        device_num = self.config.get("device", "cuda:0")
        device = torch.device(device_num)
        horizon = self.config.get("horizon", 3)
        window = self.config.get("seq_in_len", 168)
        normalize = self.config.get("normalize", 2)

        self.P = window
        self.h = horizon

        names = file_name.split('.')
        postfix = names[-1]
        if postfix == 'txt':
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif postfix == 'h5':
            f = h5py.File(file_name, "r")
            self.rawdat = np.array(f["raw_data"])
            seq_len, num_nodes, feature_dim = self.rawdat.shape
            self.rawdat = self.rawdat.reshape(seq_len, num_nodes*feature_dim)
        elif postfix == 'csv':
            self.rawdat = pd.read_csv(file_name).values[:, 1:]

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self.scaler = self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

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

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.
        scaler = self._get_scalar(x_train=self.rawdat, y_train=self.rawdat)
        self.scale = np.max(np.abs(self.rawdat), axis=0)
        self.dat = scaler.transform(self.rawdat)
        return scaler
        #TODO
        # if (normalize == 0):
        #     self.dat = self.rawdat
        #
        # if (normalize == 1):
        #     self.dat = self.rawdat / np.max(self.rawdat)
        #
        # # normlized by the maximum value of each row(sensor).
        # if (normalize == 2):
        #     for i in range(self.m):
        #         self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
        #         self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集

        return self.train, self.valid, self.test

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {"scaler": self.scaler, "scale":self.scale, "rse": self.rse, "rae": self.rae}

        return feature