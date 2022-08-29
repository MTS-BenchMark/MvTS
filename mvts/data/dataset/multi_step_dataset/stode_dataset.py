import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from tqdm import tqdm
from fastdtw import fastdtw

from mvts.data.dataset.multi_step_dataset import MultiStepDataset


class STODEDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        
        self.sigma1 = self.config.get("sigma1", 0.1)
        self.thres1 = self.config.get("thres1", 0.6)

        self.sigma2 = self.config.get("sigma2", 10)
        self.thres2 = self.config.get("thres2", 0.5)

        # self.adj_mx = torch.FloatTensor(self._construct_adj())
        dtw_matrix, sp_matrix = self._construct_adj()
        self.dtw_mx = self._get_normalized_adj(dtw_matrix)

        self.sp_mx = self._get_normalized_adj(sp_matrix)


    def _construct_adj(self):
        
        data = self.rawdat
       
        num_node = data.shape[1]
        mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
        std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
        data = (data - mean_value) / std_value
        mean_value = mean_value.reshape(-1)[0]
        std_value = std_value.reshape(-1)[0]

        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]

        # np.save(f'mvts/raw_data/PEMS03/spatial_distance.npy', dtw_distance)
        
        # dtw_distance = np.load(f'mvts/raw_data/PEMS03/spatial_distance.npy')

        dist_matrix = dtw_distance
        # mean = np.mean(dist_matrix)
        # std = np.std(dist_matrix)
        # dist_matrix = (dist_matrix - mean) / std
        sigma = self.sigma1
        dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
        dtw_matrix = np.zeros_like(dist_matrix)
        dtw_matrix[dist_matrix > self.thres1] = 1


        dist_matrix = self.adj_mx
        # normalization
        std = np.std(dist_matrix[dist_matrix != np.float('inf')])
        mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
        dist_matrix = (dist_matrix - mean) / std
        sigma = self.sigma2
        sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
        sp_matrix[sp_matrix < self.thres2] = 0 

        print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
        print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
        
        return dtw_matrix, sp_matrix

    
    def _get_normalized_adj(self, A):
        alpha = 0.8
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                            diag.reshape((1, -1)))
        A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
        return torch.from_numpy(A_reg.astype(np.float32))


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
            "dtw_mx": self.dtw_mx,
            "sp_mx": self.sp_mx,
            "num_batches": self.data['num_batches']
        }

        return feature









