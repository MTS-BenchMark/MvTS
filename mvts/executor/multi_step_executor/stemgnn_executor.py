import os
from datetime import datetime
import numpy as np
import torch
import copy
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model import loss
from functools import partial


class StemGNNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.evaluator = get_evaluator(config)
        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.save = self.config.get('save', "")
        self.patience = self.config.get('patience', 10)
        self.epochs = self.config.get("epochs", 100)
        self.num_nodes = self.config.get("num_nodes", 0) * self.config.get("input_dim", 1)
        self.batch_size = self.config.get("batch_size", 64)
        self.validate_freq = self.config.get("validate_freq", 1)
        self.window_size = self.config.get("window_size", 12)
        self.horizon = self.config.get("horizon", 3)
        self.normalize_method = self.config.get("normalize", 0)
        self.early_stop = self.config.get("early_stop", False)

        self.train_normalize_statistic = self.model.train_scaler
        self.valid_normalize_statistic = self.model.valid_scaler
        self.test_normalize_statistic = self.model.test_scaler

        self.optim = self.config.get('optim', "RMSprop")
        self.lr = self.config.get('lr', 0.001)
        self.weight_decay = self.config.get('decay_rate', 0.0001)
        self.exponential_decay_step = self.config.get('exponential_decay_step', 5)

        self.cache_dir = './mvts/cache/model_cache'
        self.evaluate_res_dir = './mvts/cache/evaluate_cache'
        self.summary_writer_dir = './mvts/log/runs'
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)

        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))

        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        if self.optim == 'RMSProp':
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.lr, eps=1e-08)
        else:
            self.my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.my_optim, gamma=self.weight_decay)
        self.forecast_loss = nn.MSELoss(reduction='mean').to(self.device)

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        self._logger.info(f"Total Trainable Params: {total_params}")

    def train(self, train_loader, valid_loader):
        valid_loader, test_loader = valid_loader[0], valid_loader[1]
        test_data = [test_loader, self.test_normalize_statistic]
        best_validate_mae = np.inf
        wait = 0
        self.best_model = copy.deepcopy(self.model)
        valid_data = [valid_loader, self.valid_normalize_statistic]
        performance_metrics = self.validate(valid_data)
        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()
            loss_total = 0
            cnt = 0
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                self.model.zero_grad()
                forecast, _ = self.model(inputs)
                loss = self.forecast_loss(forecast, target)
                cnt += 1
                loss.backward()
                self.my_optim.step()
                loss_total += float(loss)
            self._logger.info('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt))
            if (epoch + 1) % self.exponential_decay_step == 0:
                self.my_lr_scheduler.step()
            if (epoch + 1) % self.validate_freq == 0:
                self._logger.info('------ validate on data: VALIDATE ------')
                performance_metrics = self.validate(valid_data)
                self._logger.info('------ test on data: VALIDATE ------')
                _ = self.validate(test_data)
                if best_validate_mae > performance_metrics['mae']:
                    best_validate_mae = performance_metrics['mae']
                    wait = 0
                    self.best_model = copy.deepcopy(self.model)
                else:
                    wait += 1
            # early stop
            if self.early_stop and wait >= self.patience:
                self._logger.info("early_stop at epoch: {}".format(epoch))
                break
        self.model = copy.deepcopy(self.best_model)

    def evaluate(self, test_data):
        self._logger.info('------------begin test---------')
        dataloader, scaler = test_data, self.test_normalize_statistic
        forecast_norm, target_norm = self.inference(dataloader)
        if (self.normalize_method is not 0) and scaler:
            forecast = scaler.inverse_transform(forecast_norm)
            target = scaler.inverse_transform(target_norm)
        else:
            forecast, target = forecast_norm, target_norm
        forecast = forecast[:, :, :, None]
        target = target[:, :, :, None]
        res_scores = self.evaluator.evaluate(forecast, target)
        self._logger.info('test_escore: {}'.format(res_scores))
        # for _index in res_scores.keys():
        #     print(_index, " :")
        #     step_dict = res_scores[_index]
        #     for j, k in step_dict.items():
        #         print(j, " : ", k.item())
        # return res_scores

    def inference(self, dataloader):
        forecast_set = []
        target_set = []
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], self.horizon, self.num_nodes], dtype=np.float)
                while step < self.horizon:
                    forecast_result, a = self.model(inputs)

                    len_model_output = forecast_result.size()[1]
                    if len_model_output == 0:
                        raise Exception('Get blank inference result')
                    inputs[:, :self.window_size - len_model_output, :] = inputs[:, len_model_output:self.window_size,
                                                                    :].clone()
                    inputs[:, self.window_size - len_model_output:, :] = forecast_result.clone()
                    forecast_steps[:, step:min(self.horizon - step, len_model_output) + step, :] = forecast_result[:, :min(self.horizon - step, len_model_output), :].detach().cpu().numpy()
                    step += min(self.horizon - step, len_model_output)
                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())
        return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

    def validate(self, data): #验证
        dataloader, scaler = data[0], data[1]
        forecast_norm, target_norm = self.inference(dataloader)
        if (self.normalize_method is not 0) and scaler:
            forecast = scaler.inverse_transform(forecast_norm)
            target = scaler.inverse_transform(target_norm)
        else:
            forecast, target = forecast_norm, target_norm
        forecast = forecast[:, :, :, None]
        target = target[:, :, :, None]
        score = self.evaluator.evaluate(forecast, target)
        mape, mae, rmse = score['masked_MAPE']['all'], score['masked_MAE']['all'], score['masked_RMSE']['all']
        self._logger.info(f'MAPE {mape:7.9}; MAE {mae:7.9f}; RMSE {rmse:7.9f}.')
        return dict(mae=mae, mape=mape, rmse=rmse)

    # def de_normalized(self, data, norm_statistic):
    #     if self.normalize_method == 2:
    #         if not norm_statistic:
    #             norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
    #         scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
    #         data = data * scale + norm_statistic['min']
    #     elif self.normalize_method == 1:
    #         if not norm_statistic:
    #             norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
    #         mean = norm_statistic['mean']
    #         std = norm_statistic['std']
    #         std = [1 if i == 0 else i for i in std]
    #         data = data * std + mean
    #     return data


    def save_model(self, cache_name):
        """
        将当前的模型保存到文件
        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at {}".format(cache_name))
        torch.save(self.best_model.state_dict(), cache_name)


    def load_model(self, cache_name):
        """
        加载对应模型的 cache
        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at {}".format(cache_name))
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)