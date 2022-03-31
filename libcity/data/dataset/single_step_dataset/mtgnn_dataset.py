import torch
import numpy as np
from torch.autograd import Variable
from libcity.data.dataset import AbstractDataset

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class MTGNNDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        file_name = self.config.get("filename", " ")
        train = self.config.get("train_rate", 0.6)

        valid = self.config.get("eval_rate", 0.2)
        device_num = self.config.get("device", "cuda:0")
        device = torch.device(device_num)
        horizon = self.config.get("horizon", 3)
        window = self.config.get("seq_in_len", 168)
        normalize = self.config.get("normalize", 2)

        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

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

    # def get_batches(self, inputs, targets, batch_size, shuffle=True):
    #     length = len(inputs)
    #     if shuffle:
    #         index = torch.randperm(length)
    #     else:
    #         index = torch.LongTensor(range(length))
    #     start_idx = 0
    #     while (start_idx < length):
    #         end_idx = min(length, start_idx + batch_size)
    #         excerpt = index[start_idx:end_idx]
    #         X = inputs[excerpt]
    #         Y = targets[excerpt]
    #         X = X.to(self.device)
    #         Y = Y.to(self.device)
    #         yield Variable(X), Variable(Y)
    #         start_idx += batch_size

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
        feature = {"scaler": self.scale, "rse": self.rse, "rae": self.rae}

        return feature