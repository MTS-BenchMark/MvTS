import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import torch
from torch.autograd import Variable

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import DataLoader, load_pickle, DataLoaderM_new
from libcity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir

from libcity.data.dataset.multi_step_dataset import MultiStepDataset

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(
        np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class DGCRNDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        del self.data
        self.strides = self.config.get("strides", 3)
        adj_mx = self._construct_adj()
        device = torch.device(config.get("device", "cpu"))
        self.adj_mx = [torch.tensor(adj).to(device) for adj in adj_mx]
        self.data = self._gene_dataset()


    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        adj_mx = self.adj_mx

        return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]


    def _gene_dataset(self):
        data = {}
        self.train, self.valid, self.test = self._generate_train_val_test()
        x_train, y_train = self.train[0], self.train[1]
        x_valid, y_valid = self.valid[0], self.valid[1]
        x_test, y_test = self.test[0], self.test[1]
        self.scaler = self._get_scalar(x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        x_train[..., :self.output_dim] =  self.scaler.transform(x_train[..., :self.output_dim])
        x_valid[..., :self.output_dim] =  self.scaler.transform(x_valid[..., :self.output_dim])
        x_test[..., :self.output_dim] =  self.scaler.transform(x_test[..., :self.output_dim])
        import copy
        y_train_cl = copy.deepcopy(y_train)
        y_train_cl[..., :self.output_dim] =  self.scaler.transform(y_train_cl[..., :self.output_dim])

        data['train_loader'] = DataLoaderM_new(x_train, y_train,
                                          y_train_cl, self.batch_size)
        data['valid_loader'] = DataLoader(x_valid, y_valid,
                                          self.batch_size)
        data['test_loader'] = DataLoader(x_test, y_test, self.batch_size)
        data['scaler'] = self.scaler
        data['num_batches'] = x_train.shape[0] / self.batch_size
        return data


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
            "num_batches": self.data['num_batches']
        }

        return feature









