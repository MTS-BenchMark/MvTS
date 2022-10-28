import torch
import os
import h5py
import numpy as np
import pandas as pd
import pickle
import csv
from torch.autograd import Variable
from mvts.data.dataset import AbstractDataset
from torch.utils.data import TensorDataset, DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

# class MaxMinScaler:
#     def __init__(self, _min, _max):
#         self._min = _min
#         self._max = _max
#
#     def transform(self, x):
#         x = 1. * (x - self._min) / (self._max - self._min)
#         x = x * 2. - 1.
#         return x
#
#     def inverse_transform(self, x):
#         x = (x + 1.) / 2.
#         x = 1. * x * (self._max - self._min) + self._min
#         return x

def search_data(sequence_length, num_of_depend, label_start_idx,
                len_for_input, num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            # print('start_idx: ', start_idx)
            # print('end_idx: ', end_idx)
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None
    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, 0, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, 0, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        len_for_input = points_per_hour * num_of_hours
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, len_for_input, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    # print('hour_sample.shape: ', hour_sample.shape)
    # exit()
    return week_sample, day_sample, hour_sample, target


def MinMaxnormalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm

def read_and_generate_dataset_encoder_decoder(filename, num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict, train_ratio, valid_ratio, points_per_hour=12):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    suffix = filename.split('.')
    suffix = suffix[len(suffix)-1]
    if suffix == "npz":
        data_seq = np.load(filename)['data']  # (sequence_length, num_of_vertices, num_of_features)
    elif suffix == "h5":
        f = h5py.File(filename, "r")
        data_seq = np.array(f["raw_data"])

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * train_ratio)
    split_line2 = int(len(all_samples) * (train_ratio + valid_ratio))

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])

    return all_data

#TODO
# def load_graphdata_normY_channel1(all_data, DEVICE, batch_size, shuffle=True, percent=1.0):
#     '''
#     将x,y都处理成归一化到[-1,1]之前的数据;
#     每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
#     该函数会把hour, day, week的时间串起来；
#     注： 从文件读入的数据，x,y都是归一化后的值
#     :param graph_signal_matrix_filename: str
#     :param num_of_hours: int
#     :param num_of_days: int
#     :param num_of_weeks: int
#     :param DEVICE:
#     :param batch_size: int
#     :return:
#     three DataLoaders, each dataloader contains:
#     test_x_tensor: (B, N_nodes, in_feature, T_input)
#     test_decoder_input_tensor: (B, N_nodes, T_output)
#     test_target_tensor: (B, N_nodes, T_output)
#
#     '''
#     train_x = all_data['train']['x']  # (10181, 307, 3, 12)
#     train_x = train_x[:, :, 0:1, :]
#     train_target = all_data['train']['target']  # (10181, 307, 12)
#     train_timestamp = all_data['train']['timestamp']  # (10181, 1)
#
#     train_x_length = train_x.shape[0]
#     scale = int(train_x_length * percent)
#     print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
#     train_x = train_x[:scale]
#     train_target = train_target[:scale]
#     train_timestamp = train_timestamp[:scale]
#
#     val_x = all_data['val']['x']
#     val_x = val_x[:, :, 0:1, :]
#     val_target = all_data['val']['target']
#     val_timestamp = all_data['val']['timestamp']
#
#     test_x = all_data['test']['x']
#     test_x = test_x[:, :, 0:1, :]
#     test_target = all_data['test']['target']
#     test_timestamp = all_data['test']['timestamp']
#
#     _max = all_data['stats']['_max']  # (1, 1, 3, 1)
#     _min = all_data['stats']['_min']  # (1, 1, 3, 1)
#
#     # 统一对y进行归一化，变成[-1,1]之间的值
#     scaler = MaxMinScaler(_max=_max[:, :, 0, :], _min=_min[:, :, 0, :])
#     train_target_norm = scaler.transform(train_target)
#     test_target_norm = scaler.transform(test_target)
#     val_target_norm = scaler.transform(val_target)
#
#     # train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
#     # test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
#     # val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])
#
#     #  ------- train_loader -------
#     train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
#     train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
#     train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
#
#     train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#     train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#
#     train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     #  ------- val_loader -------
#     val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
#     val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
#     val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
#
#     val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#     val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#
#     val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
#
#     #  ------- test_loader -------
#     test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
#     test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
#     test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
#
#     test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#     test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
#
#     test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
#
#     # print
#     print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
#     print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
#     print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())
#
#     return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, scaler

def get_adjacency_matrix(distances_file, num_nodes, graph_sensor_ids, direction=1):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    A = np.zeros((int(num_nodes), int(num_nodes)),
                 dtype=np.float32)
    distaneA = np.zeros((int(num_nodes), int(num_nodes)),
                        dtype=np.float32)

    # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
    if graph_sensor_ids:
        with open(graph_sensor_ids, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split(','))}  # 把节点id（idx）映射成从0开始的索引

        with open(distances_file, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
                if direction == 2:
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[j], id_dict[i]] = distance
        return A, distaneA

    else:  # distance file中的id直接从0开始
        with open(distances_file, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = distance
                if direction == 2:
                    A[j, i] = 1
                    distaneA[j, i] = distance
        return A, distaneA


class ASTGCNDataset(AbstractDataset):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self,config):
        self.config = config
        filename = self.config.get("filename", "")
        graph_sensor_ids = self.config.get("graph_sensor_ids", None)
        distances_file = self.config.get("distances_file", "")
        train_ratio = self.config.get("train_rate", 0.7)
        eval_ratio = self.config.get("eval_rate", 0.1)
        valid_ratio = 1 - train_ratio - eval_ratio
        device_num = self.config.get("device", "cpu")
        direction = self.config.get('direction', 2)
        device = torch.device(device_num)

        points_per_hour = self.config.get('points_per_hour', 12)
        horizon = self.config.get('horizon', 12)
        num_nodes = self.config.get('num_nodes', 207)
        num_of_weeks = self.config.get('num_of_weeks', 0)
        num_of_days = self.config.get('num_of_days', 0)
        num_of_hours = self.config.get('num_of_hours', 1)
        batch_size = self.config.get("batch_size", 16)
        self.normalize = self.config.get("normalize", 0)

        all_data = read_and_generate_dataset_encoder_decoder(filename, num_of_weeks, num_of_days,
                                                             num_of_hours, horizon, train_ratio, valid_ratio, points_per_hour=points_per_hour)
        self.train_loader, self.train_target_tensor, self.val_loader, self.val_target_tensor, self.test_loader, self.test_target_tensor, self.scaler = self.load_graphdata_normY_channel1(
            all_data, device, batch_size)

        # ASTGCN的adj和DCRNN不一样，因此不保存，每次重新生成
        f = h5py.File(filename, "r")
        adj = np.array(f["adjacency_matrix"])
        adj[adj != 0.] = 1.
        self.adj_mx = adj
        # self.adj_mx, _ = get_adjacency_matrix(distances_file, num_nodes, graph_sensor_ids, direction)

    def _get_scalar(self, x_train, y_train=None):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        _max = x_train['stats']['_max']  # (1, 1, 3, 1)
        _min = x_train['stats']['_min']  # (1, 1, 3, 1)

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
            scaler = MinMax11Scaler(maxx=_max[:, :, 0, :], minn=_min[:, :, 0, :])
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

    def load_graphdata_normY_channel1(self, all_data, DEVICE, batch_size, shuffle=True, percent=1.0):
        '''
        将x,y都处理成归一化到[-1,1]之前的数据;
        每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
        该函数会把hour, day, week的时间串起来；
        注： 从文件读入的数据，x,y都是归一化后的值
        :param graph_signal_matrix_filename: str
        :param num_of_hours: int
        :param num_of_days: int
        :param num_of_weeks: int
        :param DEVICE:
        :param batch_size: int
        :return:
        three DataLoaders, each dataloader contains:
        test_x_tensor: (B, N_nodes, in_feature, T_input)
        test_decoder_input_tensor: (B, N_nodes, T_output)
        test_target_tensor: (B, N_nodes, T_output)

        '''
        train_x = all_data['train']['x']  # (10181, 307, 3, 12)
        train_x = train_x[:, :, 0:1, :]
        train_target = all_data['train']['target']  # (10181, 307, 12)
        train_timestamp = all_data['train']['timestamp']  # (10181, 1)

        train_x_length = train_x.shape[0]
        scale = int(train_x_length * percent)
        print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
        train_x = train_x[:scale]
        train_target = train_target[:scale]
        train_timestamp = train_timestamp[:scale]

        val_x = all_data['val']['x']
        val_x = val_x[:, :, 0:1, :]
        val_target = all_data['val']['target']
        val_timestamp = all_data['val']['timestamp']

        test_x = all_data['test']['x']
        test_x = test_x[:, :, 0:1, :]
        test_target = all_data['test']['target']
        test_timestamp = all_data['test']['timestamp']

        _max = all_data['stats']['_max']  # (1, 1, 3, 1)
        _min = all_data['stats']['_min']  # (1, 1, 3, 1)

        # 统一对y进行归一化，变成[-1,1]之间的值
        # scaler = MaxMinScaler(_max=_max[:, :, 0, :], _min=_min[:, :, 0, :])
        scaler = self._get_scalar(x_train=all_data)
        train_target_norm = scaler.transform(train_target)
        test_target_norm = scaler.transform(test_target)
        val_target_norm = scaler.transform(val_target)

        # train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
        # test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
        # val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

        #  ------- train_loader -------
        train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
        train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]),
                                             axis=2)  # (B, N, T)

        train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
        train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(
            DEVICE)  # (B, N, T)
        train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        #  ------- val_loader -------
        val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
        val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

        val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
        val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

        val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        #  ------- test_loader -------
        test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
        test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]),
                                            axis=2)  # (B, N, T)

        test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
        test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

        test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        # print
        print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
        print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
        print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

        return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, scaler

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

        return [self.train_loader, self.train_target_tensor], \
               [self.val_loader, self.val_target_tensor], \
               [self.test_loader, self.test_target_tensor]

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {"scaler": self.scaler,
                   "adj_mx": self.adj_mx}

        return feature