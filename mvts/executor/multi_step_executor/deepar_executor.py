import os
from datetime import datetime
import numpy as np
import copy
import torch
from tqdm import tqdm
import time
import math
from scipy.stats import pearsonr
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)

def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]

def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation, torch.sum(zero_index).item()]

def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics

def update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, predict_start, samples=None, relative=False):
    raw_metrics['ND'] = raw_metrics['ND'] + accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + accuracy_RMSE(sample_mu, labels[:, predict_start:], relative=relative)
    input_time_steps = input_mu.numel()
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [
        loss_fn(input_mu.cpu(), input_sigma.cpu(), labels[:, :predict_start].cpu()) * input_time_steps, input_time_steps]
    if samples is not None:
        raw_metrics['rou90'] = raw_metrics['rou90'] + accuracy_ROU(0.9, samples, labels[:, predict_start:], relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + accuracy_ROU(0.5, samples, labels[:, predict_start:], relative=relative)
    return raw_metrics

def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
                raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]
    return summary_metric

class DeepARExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        # self.evaluator = Evaluator(self.config, "multi_step")
        self.model = model.to(self.device)

        self.learning_rate = self.config.get("learning_rate")
        self.epochs = self.config.get("epochs")
        self.sampling = self.config.get("sampling")
        self.train_window = self.config.get("train_window")
        self.test_predict_start = self.config.get("test_predict_start")
        self.relative_metrics = self.config.get("relative_metrics")
        self.patience = self.config.get("patience")
        self.early_stop = self.config.get("early_stop")


        self.cache_dir = './mvts/cache1/model_cache'
        self.evaluate_res_dir = './mvts/cache1/evaluate_cache'
        self.summary_writer_dir = './mvts/log1/runs'
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

        # self.loss = nn.MSELoss().to(self.device)  # 定义损失函数
        self.loss = loss_fn
        self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器，传入所有网络参数

    def _train(self, epoch, train_loader, valid_loader):
        self.model.train()
        loss_epoch = np.zeros(len(train_loader))
        for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
            self.my_optim.zero_grad()
            batch_size = train_batch.shape[0]

            train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(self.device)  # not scaled
            labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(self.device)  # not scaled
            idx = idx.unsqueeze(0).to(self.device)

            loss = torch.zeros(1, device=self.device)
            hidden = self.model.init_hidden(batch_size)
            cell = self.model.init_cell(batch_size)

            for t in range(self.train_window):
                # if z_t is missing, replace it by output mu from the last time step
                zero_index = (train_batch[t, :, 0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    train_batch[t, zero_index, 0] = mu[zero_index]
                mu, sigma, hidden, cell = self.model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
                loss += self.loss(mu, sigma, labels_batch[t])

            loss.backward()
            self.my_optim.step()
            loss = loss.item() / self.train_window  # loss per timestep
            loss_epoch[i] = loss
            if i % 100 == 0:
                self._logger.info(f'epoch:{epoch}, iter:{i}, train_loss: {loss}')
        return loss_epoch


    def train(self, train_data, valid_data):
        self.trainData = train_data
        self.validData = valid_data

        min_val_ND = float('inf')
        wait = 0
        train_len = len(train_data)
        ND_summary = np.zeros(self.epochs)
        loss_summary = np.zeros((train_len * self.epochs))

        for epoch_num in range(0, self.epochs):
            self._logger.info('Epoch {}/{}'.format(epoch_num + 1, self.epochs))
            loss_summary[epoch_num * train_len:(epoch_num + 1) * train_len] = self._train(epoch_num, train_data, valid_data)
            val_metrics = self.validate(valid_data, sample=self.sampling)
            ND_summary[epoch_num] = val_metrics['ND']
            self._logger.info(f'epoch:{epoch_num}, valid_ND: {ND_summary[epoch_num]}')
            if ND_summary[epoch_num] < min_val_ND:
                min_val_ND = ND_summary[epoch_num]
                wait = 0
                self.best_model = copy.deepcopy(self.model)
                self._logger.info('++++++++++++save model+++++++++++')
            else:
                wait += 1
                self._logger.info('earlystop {} out of {}'.format(wait, self.patience))

            if self.early_stop:
                if wait >= self.patience:
                    self._logger.info('early stop!')
                    break


    def validate(self, data_loader, sample):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model.eval()
            raw_metrics = init_metrics(sample=sample)
            for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(data_loader)):
                test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(self.device)
                id_batch = id_batch.unsqueeze(0).to(self.device)
                v_batch = v.to(torch.float32).to(self.device)
                labels = labels.to(torch.float32).to(self.device)
                batch_size = test_batch.shape[1]
                input_mu = torch.zeros(batch_size, self.test_predict_start, device=self.device)  # scaled
                input_sigma = torch.zeros(batch_size, self.test_predict_start, device=self.device)  # scaled
                hidden = self.model.init_hidden(batch_size)
                cell = self.model.init_cell(batch_size)

                for t in range(self.test_predict_start):
                    # if z_t is missing, replace it by output mu from the last time step
                    zero_index = (test_batch[t, :, 0] == 0)
                    if t > 0 and torch.sum(zero_index) > 0:
                        test_batch[t, zero_index, 0] = mu[zero_index]

                    mu, sigma, hidden, cell = self.model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
                    input_mu[:, t] = v_batch[:, 0] * mu + v_batch[:, 1]
                    input_sigma[:, t] = v_batch[:, 0] * sigma

                if sample:
                    samples, sample_mu, sample_sigma = self.model.test(test_batch, v_batch, id_batch, hidden, cell,
                                                                  sampling=True)
                    raw_metrics = update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels,
                                                       self.test_predict_start, samples,
                                                       relative=self.relative_metrics)
                else:
                    sample_mu, sample_sigma = self.model.test(test_batch, v_batch, id_batch, hidden, cell)
                    raw_metrics = update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels,
                                                       self.test_predict_start, relative=self.relative_metrics)
            summary_metric = final_metrics(raw_metrics, sampling=sample)
            metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
            self._logger.info('- Full test metrics: ' + metrics_string)
            return summary_metric


    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        if self.saveflag:
            self.load_model(self.cache_dir)
        self.validate(test_data, sample=self.sampling)



    def save_model(self, cache_name):
        """
        将当前的模型保存到文件
        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.best_model.state_dict(), cache_name)
        self.cache_dir = cache_name
        self.saveflag = True


    def load_model(self, cache_name):
        """
        加载对应模型的 cache
        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)

