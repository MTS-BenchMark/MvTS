import os
from datetime import datetime
import numpy as np
import torch
from logging import getLogger
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim


class BHTARIMAExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.evaluator = Evaluator(self.config)

        self.cache_dir = './mvts/cache/model_cache'
        self.evaluate_res_dir = './mvts/cache/evaluate_cache'
        self.summary_writer_dir = './mvts/log/runs'
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)


    def train(self, train_data, valid_data):
        ts, label = train_data[0], train_data[1]
        result, _ = self.model.run(ts)
        pred = result[..., -1]
        # print('result.shape: ', result.shape) [228, 40]
        # print('pred.shape: ', pred.shape) [228, ]
        # print extracted forecasting result and evaluation indexes
        escore = self.evaluator.evaluate(pred, label)
        myACC = escore['ACC']['all']
        myRMSE = escore['RMSE']['all']
        myNRMSE = escore['NRMSE']['all']
        myND = escore['ND']['all']
        mySMAPE = escore['SMAPE']['all']
        print('Evaluation index, acc:{}, rmse:{}, nrmse:{}, nd:{}, smape:{}'.format(myACC, myRMSE, myNRMSE, myND, mySMAPE))


    def evaluate(self, test_data):
        print()

    def save_model(self, cache_name):
        print()

    def load_model(self, cache_name):
        print()

    # def save_model(self, cache_name):
    #     """
    #     将当前的模型保存到文件
    #     Args:
    #         cache_name(str): 保存的文件名
    #     """
    #     ensure_dir(self.cache_dir)
    #     self._logger.info("Saved model at " + cache_name)
    #     torch.save(self.model.state_dict(), cache_name)
    #
    # def load_model(self, cache_name):
    #     """
    #     加载对应模型的 cache
    #     Args:
    #         cache_name(str): 保存的文件名
    #     """
    #     self._logger.info("Loaded model at " + cache_name)
    #     model_state = torch.load(cache_name)
    #     self.model.load_state_dict(model_state)

