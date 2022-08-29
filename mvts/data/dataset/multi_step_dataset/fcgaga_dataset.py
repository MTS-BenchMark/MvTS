import torch
import pandas as pd
import h5py
import numpy as np
from mvts.data.dataset import AbstractDataset
from torch.utils.data import TensorDataset, DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

# class StandardScaler:
#     """
#     Standard the input
#     """
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def transform(self, data):
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean

def generate_graph_seq2seq_io_data(
        data, time, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, dim = data.shape
    data_list = [data]


    if add_time_in_day:
        time_ind = []
        for i in range(len(time)):
            time0 = np.datetime64(time[i])
            time_ind.append((time0 - time0.astype("datetime64[D]"))/np.timedelta64(1, "D"))
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, pd.to_datetime(time).dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
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

def generate_train_val_test(filename, train_ratio, valid_ratio, window, horizon):
    f = h5py.File(filename, "r")
    data = np.array(f["raw_data"])
    num_samples, num_nodes, dim = data.shape
    zero_mask = (data > 0).astype(np.float32)
    data = data.reshape(num_samples, num_nodes * dim)
    data = pd.DataFrame(data)
    # data[data == 0] = np.nan
    data = data.replace(0, np.nan)
    data = data.fillna(method='ffill')
    data = data.fillna(0.0)
    time = np.array(f["time"])
    t = []
    for i in range(time.shape[0]):
        t.append(time[i].decode())
    time = np.stack(t, axis=0)

    data = np.array(data)
    data = data.reshape(num_samples, num_nodes, dim)

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(1-window, 1, 1),)) # -11, -5, -2
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 1+horizon, 1)) # 4, 7, 13
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        data,
        time,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    x_mask, y_mask = generate_graph_seq2seq_io_data(
        zero_mask,
        time,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_val = round(num_samples * valid_ratio)
    num_train = round(num_samples * train_ratio)
    num_test = num_samples - num_val - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train] * y_mask[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val] * y_mask[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:] * y_mask[-num_test:]
    return [x_train, y_train], [x_val, y_val], [x_test, y_test]
    # for cat in ["train", "val", "test"]:
    #     _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    #     print(cat, "x: ", _x.shape, "y:", _y.shape)
    #     np.savez_compressed(
    #         os.path.join(args.output_dir, "%s-history-%d-horizon-%d.npz" % (cat, args.history_length, args.horizon)),
    #         x=_x,
    #         y=_y,
    #         x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
    #         y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    #     )

class FCGAGADataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        file_name = self.config.get("filename", "")
        train_ratio = self.config.get("train_rate", 0.7)
        valid_ratio = self.config.get("eval_rate", 0.1)
        horizon = self.config.get("horizon", 3)
        batch_size = self.config.get("batch_size", 16)
        window = self.config.get("window_size", 12)
        num_nodes = self.config.get("num_nodes", 207)
        self.normalize = self.config.get("normalize", 1)
        self.output_dim = self.config.get("output_dim", 1)

        self.P = window
        self.h = horizon
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        self.train_data, self.valid_data, self.test_data = generate_train_val_test(file_name, train_ratio, valid_ratio, self.P, self.h)
        self.scaler = self.normalized()

        self.num_batches = self.train_data[0].shape[0] / batch_size

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
        scaler = self._get_scalar(x_train=self.train_data[0][..., :self.output_dim], y_train=self.train_data[1][..., :self.output_dim])
        # scaler = StandardScaler(mean=self.train_data[0][..., 0].mean(), std=self.train_data[0][..., 0].std())
        self.train_data[0][..., :self.output_dim] = scaler.transform(self.train_data[0][..., :self.output_dim])
        self.train_data[1][..., :self.output_dim] = scaler.transform(self.train_data[1][..., :self.output_dim])
        self.valid_data[0][..., :self.output_dim] = scaler.transform(self.valid_data[0][..., :self.output_dim])
        self.valid_data[1][..., :self.output_dim] = scaler.transform(self.valid_data[1][..., :self.output_dim])
        self.test_data[0][..., :self.output_dim] = scaler.transform(self.test_data[0][..., :self.output_dim])
        self.test_data[1][..., :self.output_dim] = scaler.transform(self.test_data[1][..., :self.output_dim])
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

        return self.train_loader,  [self.valid_loader, self.test_loader],  self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        # node_id = torch.from_numpy(np.arange(self.num_nodes)).unsqueeze(0).repeat(self.batch_size, 1)
        node_id = torch.from_numpy(np.arange(self.num_nodes*self.output_dim))
        feature = {"scaler": self.scaler,
                   "num_batches": self.num_batches,
                   "node_id": node_id}

        return feature
