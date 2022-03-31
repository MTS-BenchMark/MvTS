import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
import tqdm

from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.executor.utils import get_train_loss
from libcity.utils import get_evaluator, ensure_dir, Optim
from libcity.model import loss
from functools import partial


class NBeatsExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.evaluator = get_evaluator(config)

        _device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(_device)
        self.model = model

        self.cache_dir = './libcity/cache/model_cache'
        self.evaluate_res_dir = './libcity/cache/evaluate_cache'
        self.summary_writer_dir = './libcity/log/runs'
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)

        self.train_loss = self.config.get("train_loss", "masked_mae")
        self.criterion = get_train_loss(self.train_loss) 

        self.cuda = self.config.get("cuda", True)
        self.best_val = 10000000

        self.epochs = self.config.get("epochs", 100)
        self.scaler = self.model.scaler
        self.num_batches = self.model.num_batches
        self.num_nodes = self.config.get("num_nodes", 0)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 20)
        self.lr_decay = self.config.get("lr_decay", False)
        self.window = self.config.get("window", 12)
        self.horizon = self.config.get("horizon", 12)

        self.hidden_layer_units = self.config.get("hidden_layer_units", 64)
        self.nb_blocks_per_stack = self.config.get("nb_blocks_per_stack", 2)
        self.share_weights_in_stack = self.config.get("share_weights_in_stack", True)

        self.mask = self.config.get("mask", True)
        self.models = []
        


    def train(self, train_data, valid_data):
        print("begin training")

        x_train, y_train, x_valid, y_valid = 0, 0, 0, 0
        
        for iter, (x,y) in enumerate(train_data.get_iterator()):
            if isinstance(x_train, int):
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate((x_train, x), 0)
                y_train = np.concatenate((y_train, y), 0)
        
        for iter, (x,y) in enumerate(valid_data.get_iterator()):
            if isinstance(x_valid, int):
                x_valid = x
                y_valid = y
            else:
                x_valid = np.concatenate((x_valid, x), 0)
                y_valid = np.concatenate((y_valid, y), 0)

        num_nodes = x_train.shape[2]
        assert num_nodes == self.num_nodes
        for i in range(num_nodes):

            _trainX, _trainY = x_train[:, :, i, :1], y_train[:, :, i, :1]
            _validX, _validY = x_valid[:, :, i, :1], y_valid[:, :, i, :1]

            backend = NBeatsPytorch(
                backcast_length=self.window, forecast_length=self.horizon,
                stack_types=(NBeatsPytorch.GENERIC_BLOCK, NBeatsPytorch.GENERIC_BLOCK),
                nb_blocks_per_stack=self.nb_blocks_per_stack, thetas_dim=(4, 4), 
                share_weights_in_stack=self.share_weights_in_stack,
                hidden_layer_units=self.hidden_layer_units
            )

            backend.compile(loss='mae', optimizer='adam')
            backend.fit(self.scaler.inverse_transform(_trainX),
                self.scaler.inverse_transform(_trainY), 
                validation_data=(
                    self.scaler.inverse_transform(_validX), 
                    self.scaler.inverse_transform(_validY)
                    ), 
                epochs=self.epochs, batch_size=self.batch_size)
            
            self.models.append(backend)


    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        outputs = []
        realy = []

        x_test, y_test = 0, 0
        for iter, (x,y) in enumerate(test_data.get_iterator()):
            if isinstance(x_test, int):
                x_test = x
                y_test = y
            else:
                x_test = np.concatenate((x_test, x), 0)
                y_test = np.concatenate((y_test, y), 0)

        num_nodes = x_test.shape[2]
        assert num_nodes == self.num_nodes

        outputs, realy = [], []

        for i in range(num_nodes):

            _testX, _testY = x_test[:, :, i, :1], y_test[:, :, i, :1].squeeze()
            _model = self.models[i]
            predY =  _model.predict(self.scaler.inverse_transform(_testX)).squeeze()
            assert predY.shape == _testY.shape
            outputs.append(torch.from_numpy(predY).unsqueeze(-1))
            realy.append(torch.from_numpy(self.scaler.inverse_transform(_testY)).unsqueeze(-1))
        
        realy = torch.cat(realy, dim=-1)
        yhat = torch.cat(outputs, dim=-1)

        res_scores = self.evaluator.evaluate(yhat, realy)
        for _index in res_scores.keys():
            print(_index, " :")
            step_dict = res_scores[_index]
            for j, k in step_dict.items():
                print(j, " : ", k.item())
        
        

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        return

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        raise ValueError("please retrain this model")
        return
