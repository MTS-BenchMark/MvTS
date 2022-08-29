import torch
import h5py
import pandas as pd
import numpy as np
from mvts.data.dataset import AbstractDataset
from torch.utils.data import TensorDataset, DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    n_slot = day_slot
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            end = (i + offset) * day_slot + j + 1
            sta = end - n_frame
            if sta >= 0:
                tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def generate_train_val_test(filename, train_ratio, valid_ratio, test_ratio, window, horizon, n_slots):
    # data_seq = pd.read_csv(filename, header=None).values
    f = h5py.File(filename, "r")
    data_seq = np.array(f["raw_data"])
    seq_len, num_nodes, feature_dim = data_seq.shape
    data_seq = data_seq.reshape(seq_len, num_nodes*feature_dim)
    seq_len, num_nodes = data_seq.shape
    n_frame = window + horizon
    seq_train = seq_gen(train_ratio, data_seq, 0, n_frame, num_nodes, n_slots)
    seq_train = seq_train[n_frame:]
    seq_val = seq_gen(valid_ratio, data_seq, train_ratio, n_frame, num_nodes, n_slots)
    seq_test = seq_gen(test_ratio, data_seq, train_ratio + valid_ratio, n_frame, num_nodes, n_slots)

    x_train = seq_train[:, :window, :, :]
    y_train = seq_train[:, window:, :, :]

    x_val = seq_val[:, :window, :, :]
    y_val = seq_val[:, window:, :, :]

    x_test = seq_test[:, :window, :, :]
    y_test = seq_test[:, window:, :, :]
    return [x_train, y_train], [x_val, y_val], [x_test, y_test]


class STNormDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        file_name = self.config.get("filename", "")
        train_ratio = self.config.get("train_rate") #如果得到的是浮点数，需要改为整数
        valid_ratio = self.config.get("eval_rate")
        test_ratio = 1-train_ratio-valid_ratio
        train_ratio = int(test_ratio*100)
        valid_ratio = int(valid_ratio*100)
        test_ratio = int(test_ratio*100)

        horizon = self.config.get("horizon", 3)
        window = self.config.get("window_size", 16)
        batch_size = self.config.get("batch_size", 16)
        num_nodes = self.config.get("num_nodes", 336)
        n_slots = self.config.get("n_slots", 24)
        self.normalize = self.config.get("normalize", 1)

        self.P = window
        self.h = horizon
        self.batch_size = batch_size

        self.train_data, self.valid_data, self.test_data = generate_train_val_test(file_name, train_ratio, valid_ratio, test_ratio, self.P, self.h, n_slots)
        self.scaler = self.normalized()

        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.train_data[0]), torch.from_numpy(self.train_data[1])), batch_size,
            shuffle=True)
        self.valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.valid_data[0]), torch.from_numpy(self.valid_data[1])), batch_size,
            shuffle=False)
        self.test_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.test_data[0]), torch.from_numpy(self.test_data[1])), batch_size,
            shuffle=False)

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


    def normalized(self):
        scaler = self._get_scalar(self.train_data[0], self.train_data[1])
        # scaler = StandardScaler(mean=self.train_data[0][..., 0].mean(), std=self.train_data[0][..., 0].std())
        self.train_data[0] = scaler.transform(self.train_data[0])
        self.train_data[1] = scaler.transform(self.train_data[1])
        self.valid_data[0] = scaler.transform(self.valid_data[0])
        self.valid_data[1] = scaler.transform(self.valid_data[1])
        self.test_data[0] = scaler.transform(self.test_data[0])
        self.test_data[1] = scaler.transform(self.test_data[1])
        return scaler

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

        return self.train_loader,  self.valid_loader,  self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {"scaler": self.scaler}

        return feature
