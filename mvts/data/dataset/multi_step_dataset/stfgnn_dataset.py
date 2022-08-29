import os
import pandas as pd
import numpy as np
import pickle
import torch
import time
from torch.autograd import Variable

from mvts.data.dataset.multi_step_dataset import MultiStepDataset

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STFGNNDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        self.strides = self.config.get("strides", 4)
        self.order = self.config.get("order", 1)
        self.lag = self.config.get("lag", 12)
        self.period = self.config.get("period", 288)
        self.sparsity = self.config.get("sparsity", 0.01)
        self.train_rate = self.config.get("train_rate", 0.6)
        self.adj_mx = torch.FloatTensor(self._construct_adj())
        #self.adj_mx = torch.randn((1432, 1432))


    def _construct_dtw(self):

        if len(self.rawdat < 3):
            data = np.array(self.rawdat)
        else:
            data = self.rawdat[:, :, 0]

        total_day = data.shape[0] / 288
        tr_day = int(total_day * 0.6)
        n_route = data.shape[1]
        xtr = gen_data(data, tr_day, n_route)
        print(np.shape(xtr))
        T0 = 288
        T = 12
        N = n_route
        d = np.zeros([N, N])
        for i in range(N):
            for j in range(i+1,N):
                d[i,j]=compute_dtw(xtr[:,:,i],xtr[:,:,j])

        print("The calculation of time series is done!")
        dtw = d+ d.T
        n = dtw.shape[0]
        w_adj = np.zeros([n,n])
        adj_percent = 0.01
        top = int(n * adj_percent)
        for i in range(dtw.shape[0]):
            a = dtw[i,:].argsort()[0:top]
            for j in range(top):
                w_adj[i, a[j]] = 1
        
        for i in range(n):
            for j in range(n):
                if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                    w_adj[i][j] = 1
                if( i==j):
                    w_adj[i][j] = 1

        print("Total route number: ", n)
        print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(n*n))
        print("The weighted matrix of temporal graph is generated!")
        self.dtw = w_adj


    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        self._construct_dtw()
        adj_mx = construct_adj_fusion(self.adj_mx, self.dtw, self.strides)
        print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)

        return adj_mx

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









