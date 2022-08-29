import torch
import os
import re
import numpy as np
import pandas as pd
import pickle
import h5py
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
from torch.utils.data import TensorDataset, DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_adj_mat(adj_mat_file, graph_sensor_ids, distances_file):
    if os.path.exists(adj_mat_file):
        try:
            with open(adj_mat_file, 'rb') as f:
                _, _, adj_mx = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(adj_mat_file, 'rb') as f:
                _, _, adj_mx = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', adj_mat_file, ':', e)
            raise
        # with open(adj_mat_file, 'rb') as f:
        #     _, _, adj_mx = pickle.load(f)
        #     # adj_mx = pickle.load(f)
    else:
        with open(graph_sensor_ids) as f:
            sensor_ids = f.read().strip().split(',')
        distance_df = pd.read_csv(distances_file, dtype={'from': 'str', 'to': 'str'})
        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1)
        with open(adj_mat_file, 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
    return adj_mx


def generate_graph_seq2seq_io_data(
        filename, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None, flag=0
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
    mystr = re.split("[. /]", filename)
    print(mystr)
    postfix = mystr[-1]
    prefix = mystr[-2]
    if postfix == "h5":
        f = h5py.File(filename, "r")
        data = np.array(f["raw_data"])
        time = np.array(f["time"])
        t = []
        for i in range(time.shape[0]):
            t.append(time[i].decode())
        time = np.stack(t, axis=0)
    elif postfix == "npz":
        data = np.load(filename)
        data = data[data.files[0]]
        df = data.squeeze(-1)

    num_samples, num_nodes, dim = data.shape
    data_list = [data]
    if add_time_in_day:
        time_ind = []
        for i in range(len(time)):
            time0 = np.datetime64(time[i])
            time_ind.append((time0 - time0.astype("datetime64[D]")) / np.timedelta64(1, "D"))
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, pd.to_datetime(time).dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1) #[len, nodes_num, 2]
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


def generate_train_val_test(filename, train_ratio, valid_ratio, window, horizon, flag):
    # df = pd.read_hdf(filename)
    # df = h5py.File(filename, 'r')
    # data = []
    # for group in df.keys():
    #     data0 = np.expand_dims(df[group][...], axis=-1)
    #     data.append(data0)
    # data = np.concatenate(data, axis=-1)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(window*-1+1, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, horizon+1, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        filename,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
        flag=flag
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape) #[len, window/horizon, nodes_num, 2]
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    test_ratio = 1 - train_ratio - valid_ratio
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    # for cat in ["train", "val", "test"]:
    #     _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    #     print(cat, "x: ", _x.shape, "y:", _y.shape)
    #     np.savez_compressed(
    #         os.path.join('./mvts/raw_data/nyc-bike/', "%s.npz" % cat),
    #         x=_x,
    #         y=_y,
    #         x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
    #         y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    #     )
    # exit()

    return [x_train, y_train], [x_val, y_val], [x_test, y_test]


class DCRNNDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        file_name = self.config.get("filename", "")
        graph_sensor_ids = self.config.get("graph_sensor_ids", "")
        distances_file = self.config.get("distances_file", "")
        train_ratio = self.config.get("train_rate", 0.7)
        valid_ratio = self.config.get("eval_rate", 0.1)
        device_num = self.config.get("device", "cpu")
        device = torch.device(device_num)
        horizon = self.config.get("horizon", 3)
        batch_size = self.config.get("batch_size", 16)
        window = self.config.get("window", 12)
        self.normalize = self.config.get("normalize", 0)
        flag = self.config.get("flag", 0)
        self.output_dim = self.config.get("output_dim")

        self.P = window
        self.h = horizon

        self.train_data, self.valid_data, self.test_data = generate_train_val_test(file_name, train_ratio, valid_ratio, self.P, self.h, flag)

        self.scaler = self.normalized()
        #
        self.num_batches = self.train_data[0].shape[0] / batch_size
        #
        self.train_loader = DataLoader(TensorDataset(torch.from_numpy(self.train_data[0]), torch.from_numpy(self.train_data[1])), batch_size, shuffle=False)
        self.valid_loader = DataLoader(TensorDataset(torch.from_numpy(self.valid_data[0]), torch.from_numpy(self.valid_data[1])), batch_size, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(torch.from_numpy(self.test_data[0]), torch.from_numpy(self.test_data[1])), batch_size, shuffle=False)
        # # self.sampler = self.train_loader.sampler
        self.device = device
        #
        f = h5py.File(file_name, "r")
        self.adj_mx = np.array(f["adjacency_matrix"])

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

        return self.train_loader, self.valid_loader, self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {"scaler": self.scaler,
                   "adj_mx": self.adj_mx,
                   "num_batches": self.num_batches}

        return feature