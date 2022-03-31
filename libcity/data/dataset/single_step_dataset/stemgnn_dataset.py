import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from libcity.data.dataset import AbstractDataset
import torch.utils.data as torch_data

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        # if normalize_method:
        #     self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

class StemGNNDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        file_name = self.config.get("filename", "")
        train_ratio = self.config.get("train_rate", 0.6)
        valid_ratio = self.config.get("eval_rate", 0.2)
        device_num = self.config.get("device", "cpu")
        device = torch.device(device_num)
        horizon = self.config.get("horizon", 3)
        batch_size = self.config.get("batch_size", 16)
        window = self.config.get("window_size", 168)
        normalizeNum = self.config.get("normalize", 0)

        self.P = window
        self.h = horizon

        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
        elif file_name[-2:] == "h5":
            self.rawdat = pd.read_hdf(file_name).values
        else:
            raise ValueError('file_name type error!')

        train_data = self.rawdat[:int(train_ratio * len(self.rawdat))]
        valid_data = self.rawdat[int(train_ratio * len(self.rawdat)):int((train_ratio + valid_ratio) * len(self.rawdat))]
        test_data = self.rawdat[int((train_ratio + valid_ratio) * len(self.rawdat)):]
        # print('normalizeNum: ', normalizeNum)
        # print('train_data.shape: ', train_data.shape)
        # print(train_data, train_data)

        self.train_data, self.train_normalize_statistic = self.normalized(normalizeNum, train_data)
        self.valid_data, self.valid_normalize_statistic = self.normalized(normalizeNum, valid_data)
        self.test_data, self.test_normalize_statistic = self.normalized(normalizeNum, test_data)

        train_set = ForecastDataset(self.train_data, window_size=self.P, horizon=self.h,
                                    normalize_method=normalizeNum, norm_statistic=self.train_normalize_statistic)
        valid_set = ForecastDataset(self.valid_data, window_size=self.P, horizon=self.h,
                                    normalize_method=normalizeNum, norm_statistic=self.valid_normalize_statistic)
        test_set = ForecastDataset(self.test_data, window_size=self.P, horizon=self.h,
                                   normalize_method=normalizeNum, norm_statistic=self.test_normalize_statistic)
        self.train_loader = torch_data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
        self.valid_loader = torch_data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = torch_data.DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

        self.device = device

    def normalized(self, normalize, rawdata):
        if normalize == 2:
            data_min = np.min(rawdata, axis=0).tolist()
            data_max = np.max(rawdata, axis=0).tolist()
            scale = data_max - data_min + 1e-5
            norm_data = (rawdata - data_min) / scale
            norm_data = np.clip(norm_data, 0.0, 1.0)
            normalize_statistic = {"min": data_min, "max": data_max}
            return norm_data, normalize_statistic
        elif normalize == 1:
            mean = np.mean(rawdata, axis=0).tolist()
            std = np.std(rawdata, axis=0).tolist()
            std = [1 if i == 0 else i for i in std]
            norm_data = (rawdata - mean) / std
            normalize_statistic = {"mean": mean, "std": std}
            return norm_data, normalize_statistic
        else:
            print("wrong normalize number")
            return None


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

        return self.train_loader, self.valid_loader, self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {"train_normalize_statistic": self.train_normalize_statistic,
                   "valid_normalize_statistic": self.valid_normalize_statistic,
                   "test_normalize_statistic": self.test_normalize_statistic}

        return feature