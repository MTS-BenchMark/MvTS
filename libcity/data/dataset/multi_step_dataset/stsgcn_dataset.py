import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.autograd import Variable

from libcity.data.dataset.multi_step_dataset import MultiStepDataset


class STSGCNDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        self.strides = self.config.get("strides", 3)
        self.adj_mx = torch.FloatTensor(self._construct_adj())

    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        A = self.adj_mx
        steps = self.strides
        N = len(A)  # 获得行数
        adj = np.zeros((N * steps, N * steps))

        for i in range(steps):
            """对角线代表各个时间步自己的空间图，也就是A"""
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

        for i in range(N):
            for k in range(steps - 1):
                """每个节点只会连接相邻时间步的自己"""
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            """加入自回"""
            adj[i, i] = 1

        return adj

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









