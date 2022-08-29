import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

from mvts.data.dataset.multi_step_dataset import MultiStepDataset
from mvts.data.utils import DataLoader
from mvts.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


class ESGDataset(MultiStepDataset):

    def __init__(self, config):
        
        super().__init__(config)
        self.node_fea = self._get_node_fea()

    
    def _get_node_fea(self):
        train = self.config.get("train_rate", 0.6)
        num_samples = self.rawdat.shape[0]
        num_train = round(num_samples * train)
        df = self.rawdat[:num_train]
        df = df.transpose(0, 2,1)
        print(df.shape)
        scaler = StandardScaler(df.mean(), df.std())
        train_feas = scaler.transform(df).reshape([-1,df.shape[2]])
        return train_feas


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
        feature = {"scaler": self.data["scaler"], 
        "node_fea": self.node_fea,
        "num_batches": self.data['num_batches']
        }

        return feature


