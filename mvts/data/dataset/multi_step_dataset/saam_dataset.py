import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.autograd import Variable
import warnings
import h5py

from mvts.data.dataset import AbstractDataset
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import stats


class TrainSet(Dataset):
    def __init__(self, data, timeSeq, train_ins_num ,pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        #print("building datasets from %s" % path)
        self.points,self.covariates,self.dates,self.withhold_len =  self.LoadData(data, timeSeq, pred_days,covariate_num = 4)

        print('withhold_len: ',self.withhold_len)
        series_len = self.points.shape[0]     # Length of every client's electricity consumption
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        d_hour = 24
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:,i]):
                if(j>0):
                    break
            indices = range(head+1,series_len+1-(T+ self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i,index)
                v = np.sum(self.points[index:(index+self.win_len),i])/self.win_len+1
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if(count < train_ins_num):
            replace = True
        prob = np.array(self.weight)/sum(self.weight)
        self.dic_keys = np.random.choice(range(count),train_ins_num,replace= replace,p=prob)

        print("Maxmum traning instances",count, "Total train instances are: ", train_ins_num, ' Overlap: ',overlap, ' Replace: ',replace)

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if(type(idx) != int):
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len,points_size))
        try:
            train_seq[:,0] = self.points[(series_index-1):(series_index+self.win_len-1),series_id]
        except (BaseException):
            import pdb;
            pdb.set_trace()
        train_seq[:,0] = self.points[(series_index-1):(series_index+self.win_len-1),series_id]
        train_seq[:,1:] = self.covariates[series_id,series_index:(self.win_len+series_index),:]
        gt = self.points[series_index:(series_index+T),series_id]
        scaling_factor = self.weight[key]
        train_seq[:,0] = train_seq[:,0]/scaling_factor

        return (train_seq,gt,series_id, scaling_factor)

    def CalCovariate(self,input_time):
        month = input_time.month
        weekday = input_time.weekday()
        hour = input_time.hour
        return np.array([month,weekday,hour])

    def LoadData(self,data, timeSeq, pred_days,covariate_num =4): 

        points = data
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len =  pred_days*d_hour
        withhold_len = pred_len

        dates= timeSeq

        covariates = np.zeros((seq_num,data_len,covariate_num))

        for idx, date in enumerate(dates):
            covariates[:,idx,:(covariate_num-1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(points[:,i]): # For each time series. we get its first non-zero value' index.
                if(j>0):
                    break
            covariates[i,head:,covariate_num-1] = range(data_len-head)   #  Get its age feature

            for index in range(covariate_num):
                covariates[i,head:,index] = stats.zscore(covariates[i,head:,index]) # We standardize all covariates to have zero mean and unit variance.
        return (points,covariates,dates,withhold_len)


class TestSet(Dataset):
    def __init__(self, points,covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points,covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len//dec_len
        self.test_ins_num = seq_num*rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times,0,-1):
                index = series_len - enc_len - j*dec_len
                self.sample_index[count] = (i,index)  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T,points_size))

        train_seq[:(self.enc_len+1),0] = self.points[series_index-1:(series_index +self.enc_len),series_id]
        train_seq[:,1:] = self.covariates[series_id,series_index:(series_index + T),:]

        gt = self.points[series_index+self.enc_len:(series_index+T),series_id]

        scaling_factor = np.sum(self.points[series_index:(series_index + self.enc_len),series_id])/self.enc_len+1
        train_seq[:,0] = train_seq[:,0]/scaling_factor
        return (train_seq,gt,series_id, scaling_factor)


    def LoadData(self,path,pred_days,covariate_num =4):

        data = pd.read_csv(path, sep=",", index_col=0, parse_dates=True, decimal='.')
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        d_hour = 24
        pred_len =  pred_days*d_hour
        withhold_len = pred_len

        dates= data.index

        covariates = np.zeros((seq_num,data_len,covariate_num))

        for idx, date in enumerate(dates):
            covariates[:,idx,:(covariate_num-1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(points[:,i]): # For each time series. we get its first non-zero value' index.
                if(j>0):
                    break
            covariates[i,head:,covariate_num-1] = range(data_len-head)   #  Get its age feature

            for index in range(covariate_num):
                covariates[i,head:,index] = stats.zscore(covariates[i,head:,index]) # We standardize all covariates to have zero mean and unit variance.
        return (points,covariates,dates,withhold_len)


class SAAMDataset(AbstractDataset):

    def __init__(self, config):

        self.config = config
        self.file_name = self.config.get("filename", " ")
        self.adj_filename = self.config.get("adj_filename", "")
        self.graph_sensor_ids = self.config.get("graph_sensor_ids", "")
        self.distances_file = self.config.get("distances_file", "")
        self.adj_type = self.config.get("adj_type", None)

        self.train_rate = self.config.get("train_rate", 0.6)
        self.valid_rate = self.config.get("eval_rate", 0.2)
        self.cuda = self.config.get("cuda", True)

        self.horizon = self.config.get("horizon", 12)
        self.window = self.config.get("window", 12)

        self.batch_size = self.config.get("batch_size", 64)
        self.adj_mx = None
        self.input_dim = self.config.get("input_dim", 1)
        self.output_dim = self.config.get("output_dim", 1)

        self._load_origin_data(self.file_name, self.adj_filename)


        self.timeList_gene = self.config.get("timeList_gene")
        self.train_ins_num = self.config.get("train_ins_num", 500000)
        self.pred_days = self.config.get("pred_days", 7)
        self.overlap = self.config.get("overlap")
        self.enc_len = self.config.get("enc_len")
        self.dec_len = self.config.get("dec_len")
        self.seq_len = self.config.get("seq_len")
        self.v_partition = self.config.get("v_partition")

        self._gene_dataset()


    def _load_origin_data(self, file_name, adj_name):
        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
            self.adj_mx = None
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
            self.adj_mx = None
        elif file_name[-2:] == "h5":
            # self.rawdat = pd.read_hdf(file_name)
            f = h5py.File(file_name, "r")
            data = np.array(f["raw_data"])
            adj = np.array(f["adjacency_matrix"])
            
            time = np.array(f["time"])
            t = []
            for i in range(time.shape[0]):
                t.append(time[i].decode())
            time = np.stack(t, axis=0)
            time = pd.to_datetime(time)

            self.rawdat = data
            self.time = time

            if self.adj_type == "distance":
                self.adj_mx = adj
            else:
                row, col = adj.shape
                for i in range(row):
                    for j in range(i, col):
                        if adj[i][j] > 0:
                            adj[i][j] = 1
                            adj[j][i] = 1
                        else:
                            adj[i][j] = 0
                            adj[j][i] = 0
                self.adj_mx = adj

        elif file_name[-3:] == "npz":
            mid_dat = np.load(file_name)
            self.rawdat = mid_dat[mid_dat.files[0]]
            self.adj_mx = None
        else:
            raise ValueError('file_name type error!')


    def _get_timeSeq(self):

        time = []
        for i in range(self.seq_len):
            time.append(i * self.timeList_gene["time_step"])
        
        res = pd.to_datetime(
            time,
            unit=self.timeList_gene["unit"],
            origin=self.timeList_gene["origin"]
        )

        return res


    def _gene_dataset(self):
        self.dim = min(self.input_dim, self.output_dim)
        self.rawdat = self.rawdat[:, :, :self.dim].reshape(self.rawdat.shape[0], -1)
        self.timeSeq = self.time

        num_train = self.train_ins_num
        indices = list(range(num_train))
        split = int(self.v_partition* num_train)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]


        train_set = TrainSet(self.rawdat, self.timeSeq, self.train_ins_num, self.pred_days, self.overlap, self.enc_len + self.dec_len)
        test_set = TestSet(train_set.points, train_set.covariates, train_set.withhold_len, self.enc_len, self.dec_len)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=4, sampler=train_sampler)
        self.valid_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=4, drop_last=True,sampler =valid_sampler)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size,num_workers=4)

        return 

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

        return self.train_loader, self.valid_loader, self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {

        }

        return feature
