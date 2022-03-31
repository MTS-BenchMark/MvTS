import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from scipy.spatial.distance import cdist

from libcity.data.dataset.multi_step_dataset import MultiStepDataset

def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(d,d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)


def random_walk_matrix(w) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.eye(d_inv.shape[0]) * d_inv
    return d_mat_inv.dot(w)

class Standard(object):

    def __init__(self):
        pass

    def fit(self, X):
        self.std = np.std(X)
        self.mean = np.mean(X)
        print("std:", self.std, "mean:", self.mean)

    def transform(self, X):
        X = 1. * (X - self.mean) / self.std
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = X * self.std + self.mean
        return X

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

    def rmse_transform(self, X):
        X = X * self.std
        return X
    def mae_transform(self, X):
        X = X* self.std
        return X


class CCRNNDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        self._len = self.config.get("_len", [672, 672]) 
        self.pre_hidden_size = self.config.get("pre_hidden_size", 20)
        self.method = self.config.get("method", "big")
        self.normalized_category = self.config.get("normalized_category", "randomwalk")
        self.support = self.preprocessing_for_metric()
        self.matrix = self.graph_preprocess()


    def preprocessing_for_metric(self):
        # normal_method = Standard()
        # data = []
        # data.append(normal_method.fit_transform(self.rawdat))
        # data = np.concatenate(data, axis=1).transpose((0,2,1))
        data = self.scaler.transform(self.rawdat).transpose((0,2,1))
        data = data[:-(self._len[0]+self._len[1])]
        T, input_dim, N = data.shape
        inputs = data.reshape(-1, N)
        u, s, v = np.linalg.svd(inputs)
        w = np.diag(s[:self.pre_hidden_size]).dot(v[:self.pre_hidden_size,:]).T
        support = None
        if self.method == 'big':
            graph = cdist(w, w, metric='euclidean')
            # support = cdist(w, w, metric='correlation')
            # support[support<=0.75] = 0
            # support[support>0.75] = 1
            # s,v,d = np.linalg.svd(graph)
            # print(v)
            support = graph * -1 / np.std(graph) ** 2
            support = np.exp(support)
            # s,v,d = np.linalg.svd(support)
            # print(support)
            # print(v)
        elif self.method == 'small':
            support = w
            print(w.shape)
        return support

    
    def graph_preprocess(self):
        matrix = self.support
        matrix = matrix - np.identity(matrix.shape[0])
        if self.normalized_category == 'randomwalk':
            matrix = random_walk_matrix(matrix)
        elif self.normalized_category == 'laplacian':
            matrix = normalized_laplacian(matrix)
        else:
            raise KeyError()
        return matrix


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {
            "scaler": self.data["scaler"],
            "adj_mx": self.adj_mx,
            "num_batches": self.data['num_batches'],
            "support": self.support,
            "matrix": self.matrix
        }

        return feature









