import torch
import numpy as np
import pandas as pd
import h5py
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
import torch.utils.data as torch_data
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

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
        target_data = self.data[hi+self.horizon-1: hi+self.horizon]
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
        self.normalize = self.config.get("normalize", 0)

        self.P = window
        self.h = horizon

        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values[:, 1:]
            self.rawdat = np.float32(self.rawdat)
            # self.rawdat = pd.read_csv(file_name).values
        elif file_name[-2:] == "h5":
            f = h5py.File(file_name, "r")
            self.rawdat = np.array(f["raw_data"])
            seq_len, num_nodes, feature_dim = self.rawdat.shape
            self.rawdat = self.rawdat.reshape(seq_len, num_nodes * feature_dim)
        else:
            raise ValueError('file_name type error!')
        print(self.rawdat)
        print(self.rawdat.shape)
        train_data = self.rawdat[:int(train_ratio * len(self.rawdat))]
        valid_data = self.rawdat[int(train_ratio * len(self.rawdat)):int((train_ratio + valid_ratio) * len(self.rawdat))]
        test_data = self.rawdat[int((train_ratio + valid_ratio) * len(self.rawdat)):]
        # print('normalizeNum: ', normalizeNum)
        # print('train_data.shape: ', train_data.shape)
        # print(train_data, train_data)

        self.train_data, self.train_scaler = self.normalized(train_data)
        self.valid_data, self.valid_scaler = self.normalized(valid_data)
        self.test_data, self.test_scaler = self.normalized(test_data)

        train_set = ForecastDataset(self.train_data, window_size=self.P, horizon=self.h,
                                    normalize_method=self.normalize, norm_statistic=self.train_scaler)
        valid_set = ForecastDataset(self.valid_data, window_size=self.P, horizon=self.h,
                                    normalize_method=self.normalize, norm_statistic=self.valid_scaler)
        test_set = ForecastDataset(self.test_data, window_size=self.P, horizon=self.h,
                                   normalize_method=self.normalize, norm_statistic=self.test_scaler)
        self.train_loader = torch_data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
        self.valid_loader = torch_data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = torch_data.DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

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
        elif self.normalize == 0:
            scaler = NoneScaler()
            print('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def normalized(self, rawdata):
        scaler = self._get_scalar(x_train=rawdata, y_train=None)
        return scaler.transform(rawdata), scaler


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
        feature = {"train_scaler": self.train_scaler,
                   "valid_scaler": self.valid_scaler,
                   "test_scaler": self.test_scaler}

        return feature