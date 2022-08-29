import os
import time
import numpy as np
import torch
import copy
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from tqdm import tqdm
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model.loss import masked_mae_torch, unmasked_mae
from functools import partial

def mae_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
    else:
        mae = np.abs(np.subtract(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())).astype('float32')
    return np.mean(mae)


def rmse_np(preds, labels):
    mse = mse_np(preds, labels)
    return np.sqrt(mse)

def mse_np(preds, labels):
    if isinstance(preds, np.ndarray):
        return np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
    else:
        return np.mean(np.square(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32'))

def mape_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    else:
        mape = np.abs(np.divide(np.subtract(preds.cpu().numpy(), labels.cpu().numpy()).astype('float32'), labels.cpu().numpy()))
    return np.mean(mape)

def corr_np(preds, labels):
    sigma_p = (preds).std(axis=0)
    sigma_g = (labels).std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = labels.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((preds - mean_p) * (labels - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation

def metric(pred, real):
    # mae = masked_mae(pred,real,0.0).item()
    # mape = masked_mape(pred,real,0.0).item()
    # rmse = masked_rmse(pred,real,0.0).item()
    mae = mae_np(pred, real)
    mape = mape_np(pred, real)
    rmse = rmse_np(pred, real)
    corr = corr_np(pred, real)
    return mae, mape, rmse, corr


class MTGNNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.evaluator = Evaluator(self.config)

        self.scaler = self.model.scaler

        self.patience = self.config.get('patience', 10)
        self.optim = self.config.get('optim')
        self.learning_rate = self.config.get('learning_rate')
        self.clip = self.config.get('clip')
        self.weight_decay = self.config.get('weight_decay')
        self.step_size1 = self.config.get('step_size1')
        self.step_size2 = self.config.get('step_size2')
        self.iter = 1
        self.task_level = 1
        self.horizon = self.config.get('horizon')
        self.cl = self.config.get('cl')
        self.epochs = self.config.get('epochs')
        self.batch_size = self.config.get('batch_size')
        self.num_nodes = self.config.get('num_nodes')
        self.num_split = self.config.get('num_split')
        self.validate_freq = self.config.get('validate_freq')
        self.output_dim = self.config.get('output_dim')

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
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def calculateLoss(self, pred, truth):
        if self.config.get('mask'):
            self.loss = masked_mae_torch
            return self.loss(pred, truth, 0.0)
        else:
            self.loss = unmasked_mae
            return self.loss(pred, truth)

    def train(self, train_loader, valid_loader):
        min_val_loss = float('inf')
        self.best_model = copy.deepcopy(self.model)
        best_epoch = -1
        wait = 0

        val_loss, val_post, _ = self.validate([valid_loader, "valid"])
        self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))

        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            total_loss = []
            for run, (x, y) in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                x = x.type(torch.FloatTensor) #TODO
                y = y.type(torch.FloatTensor)
                self.model.zero_grad()
                y = self.scaler.inverse_transform(y)
                trainx = x.to(self.device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(self.device)
                trainy = trainy.to(self.device)
                if run % self.step_size2 == 0:
                    perm = np.random.permutation(range(self.num_nodes))
                num_sub = int(self.num_nodes / self.num_split)
                for j in range(self.num_split):
                    if j != self.num_split - 1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.tensor(id).to(self.device)
                    tx = trainx[:, :, id, :]
                    ty = trainy[:, :, id, :]
                    output = self.model(tx, idx=id) #tx.shape[batch, in_dim, nodes_num, seq_len]
                    #output.shape[batch, horizon, nodes_num, output_dim]
                    predict = self.scaler.inverse_transform(output)
                    real = ty[:, :, :, :self.output_dim]

                    if self.iter % self.step_size1 == 0 and self.task_level <= self.horizon:
                        self.task_level += 1
                    if self.cl:
                        loss = self.calculateLoss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level])
                    else:
                        loss = self.calculateLoss(predict, real)
                    total_loss.append(loss.item())
                    loss.backward()

                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(total_loss):.8f}')

            if (epoch_num+1) % self.validate_freq == 0:
                val_loss, val_post, _ = self.validate([valid_loader, "valid"])
                self._logger.info('val_loss: {}'.format(val_loss))
                # test_loss, test_post, _ = self.validate([test_loader, "test"])
                # self._logger.info('test_loss: {}, test_post: {}'.format(test_loss, test_post))
                if val_loss < min_val_loss:
                    best_epoch = epoch_num
                    wait = 0
                    min_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)
                else:
                    wait += 1
            if wait >= self.patience:
                self._logger.info('early stop!')
                break
            print('best epoch is: ', best_epoch)
        self.model = copy.deepcopy(self.best_model)


    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            test_loss, test_post, escore = self.validate([test_data, "test"])
            self._logger.info('test_loss: {}'.format(test_loss))
            self._logger.info('test_escore: {}'.format(escore))


    def validate(self, dataloader):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        dataloader, type = dataloader[0], dataloader[1]
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(dataloader, ncols=100)
            losses = []
            y_truths = []
            y_preds = []
            for _, (x, y) in enumerate(pbar): #[batch, seq_len, nodes_num, input_dim]
                x = x.type(torch.FloatTensor)
                y = y.type(torch.FloatTensor)
                y = self.scaler.inverse_transform(y)
                testx = torch.Tensor(x).to(self.device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(self.device)
                output = self.model(testx) #output.shape[batch, horizon, nodes_num, output_dim]
                predict = self.scaler.inverse_transform(output)
                real = testy[:, :, :, :self.output_dim]
                loss = self.calculateLoss(predict, real)
                losses.append(loss.item())
                y_truths.append(real.cpu())
                y_preds.append(predict.cpu())
            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)

            escore = self.evaluator.evaluate(y_preds, y_truths)
            message = []
            prediction_length = y_preds.shape[1]
            assert prediction_length == self.horizon
            for i in range(prediction_length):
                mae = escore['masked_MAE'][f'horizon-{i}']
                rmse = escore['masked_RMSE'][f'horizon-{i}']
                mape = escore['masked_MAPE'][f'horizon-{i}']
                corr = escore['CORR'][f'horizon-{i}']
                log = 'Evaluate on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test CORR: {:.4f}'
                if type == "test":
                    self._logger.info(log.format(i + 1, mae, mape, rmse, corr))
                message.append("MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}， CORR: {:.4f}".format(mae, mape, rmse, corr))
            maeAll = escore['masked_MAE']['all']
            rmseAll = escore['masked_RMSE']['all']
            mapeAll = escore['masked_MAPE']['all']
            corrAll = escore['CORR']['all']
            log = 'Evaluate on test data for horizon all, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test CORR: {:.4f}'
            if type == "test":
                self._logger.info(log.format(maeAll, mapeAll, rmseAll, corrAll))
            message.append("MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}， CORR: {:.4f}".format(maeAll, mapeAll, rmseAll, corrAll))
            post_fix = {
                "type": type,
                "horizon 1": message[0],
                "horizon 2": message[1],
                "horizon 3": message[2],
                "horizon 4": message[3],
                "horizon 5": message[4],
                "horizon 6": message[5],
                "horizon 7": message[6],
                "horizon 8": message[7],
                "horizon 9": message[8],
                "horizon 10": message[9],
                "horizon 11": message[10],
                "horizon 12": message[11],
                "horizon All": message[12]
            }
            return mean_loss, post_fix, escore


    def save_model(self, cache_name):
        """
        将当前的模型保存到文件
        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.best_model.state_dict(), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache
        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)






