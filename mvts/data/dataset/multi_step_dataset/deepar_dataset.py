import torch
import os
import numpy as np
import pandas as pd
import h5py
import datetime
import pickle
import csv
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler
from scipy import stats
from tqdm import trange
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.label[index])

class TestDataset(Dataset):
    def __init__(self, data, v, label):
        self.data = data
        self.v = v
        self.label = label
        self.test_len = self.data.shape[0]

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, v, replacement=True):
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

class DeepARDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, config):
        self.config = config
        filename = self.config.get("filename")
        device_num = self.config.get("device", "cpu")
        self.device = torch.device(device_num)
        self.num_covariates = self.config.get("num_covariates")
        self.window_size = self.config.get("window_size")
        self.stride_size = self.config.get("stride_size")
        self.num_nodes = self.config.get("num_nodes")
        self.seq_len = self.config.get("seq_len")
        self.train_rate = self.config.get("train_rate")
        self.batch_size = self.config.get("batch_size")
        self.predict_batch = self.config.get("predict_batch")
        self.normalize = self.config.get("normalize", 0)

        f = h5py.File(filename, "r")
        data = np.array(f["raw_data"])
        seq_len, num_nodes, feature_dim = data.shape
        data = data.reshape(seq_len, num_nodes * feature_dim)

        adj = np.array(f["adjacency_matrix"])

        time = np.array(f["time"])
        t = []
        for i in range(time.shape[0]):
            t.append(time[i].decode())
        time = np.stack(t, axis=0)
        time = pd.to_datetime(time)
        # print(time)
        # exit()
        train_start_index = 1
        train_end_index = int(self.seq_len*self.train_rate)
        self.train_start = str(time[train_start_index])
        self.train_end = str(time[train_end_index])
        self.test_start = str(time[train_end_index]+datetime.timedelta(days=-7))
        self.test_end = str(time[-2])
        print(self.train_start)
        print(self.train_end)
        print(self.test_start)
        print(self.test_end)

        
        # print(time0)
        # exit()

        num_nodes = num_nodes * feature_dim
        indexs = ['MT_'+str(i+1) for i in range(num_nodes)]
        data2 = {indexs[i]: pd.Series(data[:, i], index=time) for i in range(num_nodes)}
        data_frame = pd.DataFrame(data2)

        # data_frame = pd.read_csv(filename, sep=";", index_col=0, parse_dates=True, decimal=',')  # 15min一条记录
        data_frame = data_frame.resample('1H', label='left', closed='right').sum()[self.train_start:self.test_end]  # 筛选，1h一条记录
        data_frame.fillna(0, inplace=True)
        # covariates是时间信息
        covariates = gen_covariates(data_frame[self.train_start:self.test_end].index, self.num_covariates)
        train_data = data_frame[self.train_start:self.train_end].values
        test_data = data_frame[self.test_start:self.test_end].values
        self.data_start = (train_data != 0).argmax(axis=0)  # find first nonzero value in each time series
        self.total_time = data_frame.shape[0]  # 32304
        self.num_series = data_frame.shape[1]  # 370



        self.train_data, self.train_v, self.train_label = self.prep_data(train_data, covariates, self.data_start, train=True)
        self.test_data, self.test_v, self.test_label = self.prep_data(test_data, covariates, self.data_start, train=False)

        scaler = self._get_scalar(x_train=self.train_data, y_train=self.train_label)
        self.train_data = scaler.transform(self.train_data)
        self.train_label = scaler.transform(self.train_label)
        self.test_data = scaler.transform(self.test_data)
        self.test_label = scaler.transform(self.test_label)

        train_set = TrainDataset(self.train_data, self.train_label)
        test_set = TestDataset(self.test_data, self.test_v, self.test_label)
        sampler = WeightedSampler(self.train_v)  # Use weighted sampler instead of random sampler
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=self.predict_batch, sampler=RandomSampler(test_set),
                                 num_workers=4)

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

    def prep_data(self, data, covariates, data_start, train=True):
        time_len = data.shape[0]
        input_size = self.window_size - self.stride_size
        windows_per_series = np.full((self.num_series), (time_len - input_size) // self.stride_size)
        if train:
            windows_per_series -= (data_start + self.stride_size - 1) // self.stride_size

        total_windows = np.sum(windows_per_series)
        x_input = np.zeros((total_windows, self.window_size, 1 + self.num_covariates + 1), dtype='float32')
        label = np.zeros((total_windows, self.window_size), dtype='float32')
        v_input = np.zeros((total_windows, 2), dtype='float32')

        count = 0
        if not train:
            covariates = covariates[-time_len:]
        for series in trange(self.num_series):
            cov_age = stats.zscore(np.arange(self.total_time - data_start[series]))
            if train:
                covariates[data_start[series]:time_len, 0] = cov_age[:time_len - data_start[series]]
            else:
                covariates[:, 0] = cov_age[-time_len:]
            for i in range(windows_per_series[series]):
                if train:
                    window_start = self.stride_size * i + data_start[series]
                else:
                    window_start = self.stride_size * i
                window_end = window_start + self.window_size
                x_input[count, 1:, 0] = data[window_start:window_end - 1, series]
                x_input[count, :, 1:1 + self.num_covariates] = covariates[window_start:window_end, :]
                x_input[count, :, -1] = series
                label[count, :] = data[window_start:window_end, series]
                nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
                if nonzero_sum == 0:
                    v_input[count, 0] = 0
                else:
                    v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum) + 1
                    x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                    if train:
                        label[count, :] = label[count, :] / v_input[count, 0]
                count += 1

        return x_input, v_input, label

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

        return self.train_loader, self.test_loader, self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """

        return None