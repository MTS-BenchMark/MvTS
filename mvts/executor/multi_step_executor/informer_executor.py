import os
from datetime import datetime
import numpy as np
import copy
import torch
from tqdm import tqdm
import time
from scipy.stats import pearsonr
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim


# def RSE(pred, true):
#     return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
#
#
# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
#     return (u / d).mean(-1)
#
#
# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))
#
#
# def MSE(pred, true):
#     return np.mean((pred - true) ** 2)
#
#
# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))
#
#
# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))
#
#
# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#
#     return mae, mse, rmse, mape, mspe

def adjust_learning_rate(optimizer, epoch, lradj, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class InformerExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        # self.evaluator = Evaluator(self.config, "multi_step")
        self.model = model.to(self.device)

        self.use_amp = self.config.get('use_amp')
        self.epochs = self.config.get('epochs')
        self.validate_freq = self.config.get('validate_freq')
        self.early_stop = self.config.get('early_stop')
        self.patience = self.config.get('patience')
        self.lradj = self.config.get('lradj')
        self.learning_rate = self.config.get('learning_rate')
        self.padding = self.config.get('padding')
        self.pred_len = self.config.get('pred_len')
        self.label_len = self.config.get('label_len')
        self.output_attention = self.config.get('output_attention')
        self.inverse = self.config.get('inverse')
        self.features = self.config.get('features')
        self.saveflag = False
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

        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))

        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.loss = nn.MSELoss().to(self.device)  # 定义损失函数
        self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器，传入所有网络参数


    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).float()
        elif self.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if self.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.features == 'MS' else 0
        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

    def train(self, train_data, valid_data):
        self.trainData, self.trainDataLoader = train_data
        valid_data, test_data = valid_data[0], valid_data[1]

        min_val_loss = float('inf')
        wait = 0
        time_now = time.time()
        train_steps = len(self.trainDataLoader)

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch_num in range(0, self.epochs):
            self.model.train()

            iter_count = 0
            losses = []
            epoch_time = time.time()
            for step, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(self.trainDataLoader): #[batch_size, seq_length, nodes_num, input_dim]
                self.model.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(
                    self.trainData, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = self.loss(pred, true)
                losses.append(loss.item())

                if (step + 1) % 100 == 0:
                    self._logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(step + 1, epoch_num + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.epochs - epoch_num) * train_steps - step)
                    self._logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.my_optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.my_optim.step()

            self._logger.info("Epoch: {} cost time: {}".format(epoch_num + 1, time.time() - epoch_time))
            train_loss = np.average(losses)
            valid_loss = self.validate(valid_data)
            test_loss = self.validate(test_data)
            self._logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch_num + 1, train_steps, train_loss, valid_loss, test_loss))

            if valid_loss < min_val_loss:
                wait = 0
                min_val_loss = valid_loss
                self.best_model = copy.deepcopy(self.model)
                self._logger.info('++++++++++++save model+++++++++++')
            else:
                wait += 1
                self._logger.info('earlystop {} out of {}'.format(wait, self.patience))

            if self.early_stop:
                if wait >= self.patience:
                    self._logger.info('early stop!')
                    break

            adjust_learning_rate(self.my_optim, epoch_num + 1, self.lradj, self.learning_rate)

    def validate(self, data):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model.eval()
            dataset, dataloader = data[0], data[1]
            losses = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                pred, true = self._process_one_batch(
                    dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = self.loss(pred.detach().cpu(), true.detach().cpu())
                losses.append(loss.item())
            loss = np.average(losses)
            self.model.train()
            return loss


    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        if self.saveflag:
            self.load_model(self.cache_dir)
        with torch.no_grad():
            self.model.eval()
            test_dataset, test_loader = test_data
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
            preds = np.array(preds)  # [89, 32, pred_len, feature_dim]
            trues = np.array(trues)  # [89, 32, pred_len, feature_dim]
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)  # [len, pred_len, feature_dim]

            # mae, mse, rmse, mape, mspe = metric(preds, trues)
            # self._logger.info('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            escore = self.evaluator.evaluate(preds, trues)
            mae = escore['MAE']['all']
            mse = (escore['RMSE']['all'])**2
            rmse = escore['RMSE']['all']
            mape = escore['masked_MAPE']['all']
            mspe = escore['MSPE']['all']
            self._logger.info('mse:{}, mae:{}'.format(mse, mae))

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

