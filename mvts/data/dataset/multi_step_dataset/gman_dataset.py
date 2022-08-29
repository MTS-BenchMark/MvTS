import torch
import os
import numpy as np
import pandas as pd
import pickle
import re
import h5py
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
from torch.utils.data import TensorDataset, DataLoader
from mvts.data.generateSE import generateSE
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
        with open(adj_mat_file, 'rb') as f:
            _, _, adj_mx = pickle.load(f)
    else:
        with open(graph_sensor_ids) as f:
            sensor_ids = f.read().strip().split(',')
        distance_df = pd.read_csv(distances_file, dtype={'from': 'str', 'to': 'str'})
        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1)
        with open(adj_mat_file, 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
    return adj_mx

def load_se_file(adj_mx, adj_file, se_file):
    if not os.path.exists(se_file):
        nodes_num = adj_mx.shape[0]
        with open(adj_file, mode='w') as f:
            for i in range(nodes_num):
                for j in range(nodes_num):
                    dis = adj_mx[i][j]
                    f.write(str(i)+" "+str(j)+" "+str(dis)+"\n")
        generateSE(adj_file, se_file)
        os.remove(adj_file)

    with open(se_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return SE



def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape #metr-la:(23990, 207)
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def generate_train_val_test(filename, train_ratio, valid_ratio, window, horizon):
    f = h5py.File(filename, "r")
    traffic = np.array(f["raw_data"])
    traffic = torch.from_numpy(traffic)
    num_step, num_nodes, feature_dim = traffic.shape
    traffic = traffic.reshape(num_step, num_nodes*feature_dim)

    time = np.array(f["time"])
    t = []
    for i in range(time.shape[0]):
        t.append(time[i].decode())
    time = np.stack(t, axis=0)
    time = pd.to_datetime(time)

    # train/val/test
    train_steps = round(train_ratio * num_step)
    val_steps = round(valid_ratio * num_step)
    test_steps = num_step - train_steps - val_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, window, horizon)
    valX, valY = seq2instance(val, window, horizon)
    testX, testY = seq2instance(test, window, horizon)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # temporal embedding
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    # timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
    #             // time.freq.delta.total_seconds()
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                // (24 * 3600)
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, window, horizon)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, window, horizon)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, window, horizon)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return [trainX, trainY, trainTE], [valX, valY, valTE], [testX, testY, testTE]


class GMANDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        file_name = self.config.get("filename", "")
        se_file = self.config.get("se_file", "")
        train_ratio = self.config.get("train_rate", 0.7)
        valid_ratio = self.config.get("eval_rate", 0.1)
        horizon = self.config.get("horizon", 3)
        batch_size = self.config.get("batch_size", 16)
        window = self.config.get("window_size", 12)
        input_dim = self.config.get("input_dim", 1)
        self.normalize = self.config.get("normalize", 0)

        self.P = window
        self.h = horizon

        f = h5py.File(file_name, "r")
        adj = np.array(f["adjacency_matrix"])

        strs = re.split("[. /]", file_name)
        se_file = "SE(" + strs[-2] + ")"
        se_file = file_name.replace(strs[-2], se_file)
        se_file = se_file.replace(strs[-1], "txt")
        adj_file = se_file.replace("SE", "adj")
        self.se = load_se_file(adj, adj_file, se_file)
        self.se = np.repeat(self.se, input_dim, axis=0)

        self.train_data, self.valid_data, self.test_data = generate_train_val_test(file_name, train_ratio, valid_ratio, self.P, self.h)

        self.scaler = self.normalized()

        # self.train_loader = DataLoader(TensorDataset(torch.from_numpy(self.train_data[0]), torch.from_numpy(self.train_data[1]), torch.from_numpy(self.train_data[2])), batch_size, shuffle=True)
        # self.valid_loader = DataLoader(TensorDataset(torch.from_numpy(self.valid_data[0]), torch.from_numpy(self.valid_data[1]), torch.from_numpy(self.valid_data[2])), batch_size, shuffle=False)
        # self.test_loader = DataLoader(TensorDataset(torch.from_numpy(self.test_data[0]), torch.from_numpy(self.test_data[1]), torch.from_numpy(self.test_data[2])), batch_size, shuffle=False)

        self.train_loader = DataLoader(TensorDataset(self.train_data[0], self.train_data[1],
                          self.train_data[2]), batch_size, shuffle=True)
        self.valid_loader = DataLoader(TensorDataset(self.valid_data[0], self.valid_data[1],
                          self.valid_data[2]), batch_size, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(self.test_data[0], self.test_data[1],
                          self.test_data[2]), batch_size, shuffle=False)


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
        self.train_data[0] = scaler.transform(self.train_data[0])
        self.valid_data[0] = scaler.transform(self.valid_data[0])
        self.test_data[0] = scaler.transform(self.test_data[0])
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
                   "SE": self.se}

        return feature