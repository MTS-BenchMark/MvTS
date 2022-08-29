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

def MAPE(v, v_):
    greater_than_20 = v > 20
    v = v[greater_than_20]
    v_ = v_[greater_than_20]
    return np.mean(np.abs(v_ - v) / (v))


def RMSE(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    return np.mean(np.abs(v_ - v))


def calculate_metrics(truths, preds):
    dim = len(preds.shape)
    if dim == 3:
        # single_step case
        return np.array([MAPE(truths, preds), MAE(truths, preds), RMSE(truths, preds)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        truths = np.swapaxes(truths, 0, 1)
        preds = np.swapaxes(preds, 0, 1)
        # recursively call
        for i in range(preds.shape[0]):
            tmp_res = calculate_metrics(truths[i], preds[i])
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)



class STNormExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        # self.evaluator = Evaluator(self.config, "multi_step")
        self.model = model.to(self.device)
        self.scaler = self.model.scaler

        self.epochs = self.config.get('epochs')
        self.horizon = self.config.get('horizon')
        self.validate_freq = self.config.get('validate_freq')
        self.early_stop = self.config.get('early_stop')
        self.patience = self.config.get('patience')
        self.learning_rate = self.config.get('learning_rate')
        self.max_grad_norm = self.config.get('max_grad_norm')
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
        print("init model...")
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        self.my_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)  # 定义优化器，传入所有网络参数


    def calculate_metrics1(self, preds, truths):
        # preds.shape: [total_time, horizon, nodes_num, feature_dim]
        escore = self.evaluator.evaluate(preds, truths)
        message = []
        prediction_length = preds.shape[1]
        #MAPE MAE RMSE
        for i in range(prediction_length):
            mae = escore['MAE'][f'horizon-{i}']
            rmse = escore['RMSE'][f'horizon-{i}']
            mape = escore['STNorm_MAPE'][f'horizon-{i}']
            message.extend([mape, mae, rmse])
        # post_fix = {
        #     "type": type,
        #     "horizon 1": message[0],
        #     "horizon 2": message[1],
        #     "horizon 3": message[2],
        #     "horizon 4": message[3],
        #     "horizon 5": message[4],
        #     "horizon 6": message[5],
        #     "horizon 7": message[6],
        #     "horizon 8": message[7],
        #     "horizon 9": message[8],
        #     "horizon 10": message[9],
        #     "horizon 11": message[10],
        #     "horizon 12": message[11]
        # }

        return message


    def train(self, train_data, valid_data):
        min_val_rmse = float('inf')
        wait = 0
        time_now = time.time()
        va = self.validate(valid_data)
        for i in range(self.horizon):
            self._logger.info(f'MAPE {va[i * 3]:7.3f}, MAE {va[i * 3 + 1]:4.3f}, RMSE {va[i * 3 + 2]:6.3f};')
        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_data, ncols=100)
            losses = []
            epoch_time = time.time()
            for step, (batch_x, batch_y) in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                self.model.zero_grad()
                pred = self.model(batch_x)
                pred = self.scaler.inverse_transform(pred)
                batch_y = self.scaler.inverse_transform(batch_y)
                loss = self.loss(pred, batch_y)
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')
            self._logger.info("Epoch: {} cost time: {} loss: {}".format(epoch_num + 1, time.time() - epoch_time, np.mean(losses)))
            train_loss = np.average(losses)

            if epoch_num % self.validate_freq == 0:
                va = self.validate(valid_data)
                self._logger.info(f'Epoch {epoch_num}:')
                for i in range(self.horizon):
                    self._logger.info(f'MAPE {va[i * 3]:7.3f}, MAE {va[i * 3 + 1]:4.3f}, RMSE {va[i * 3 + 2]:6.3f};')

                total_rmse = np.sum([va[i * 3 + 2] for i in range(self.horizon)])
                if total_rmse < min_val_rmse:
                    wait = 0
                    min_val_rmse = total_rmse
                    self.best_model = copy.deepcopy(self.model)
                    self._logger.info('++++++++++++save model+++++++++++')
                else:
                    wait += 1
                    self._logger.info('earlystop {} out of {}'.format(wait, self.patience))

                if self.early_stop:
                    if wait >= self.patience:
                        self._logger.info('early stop!')
                        break

    def validate(self, data):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        pred_list = []
        truth_list = []
        with torch.no_grad():
            self.model.eval()
            for i, (batch_x, batch_y) in enumerate(data):
                batch_x = batch_x.to(self.device).float()
                pred = self.model(batch_x)
                pred = pred.data.cpu().numpy()
                pred_list.append(pred)
                batch_y = batch_y.data.cpu().numpy()
                truth_list.append(batch_y)
            pred_array = np.concatenate(pred_list, axis=0)
            truth_array = np.concatenate(truth_list, axis=0)
            pred_array = self.scaler.inverse_transform(pred_array)
            truth_array = self.scaler.inverse_transform(truth_array)
            # print(truth_array.shape) #[num_len, horizon, num_nodes, feature_dim]
            # metrics = calculate_metrics(truth_array, pred_array)
            # print(metrics)
            metrics1 = self.calculate_metrics1(pred_array, truth_array)
            self.model.train()
            return metrics1


    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        if self.saveflag:
            self.load_model(self.cache_dir)
        result = self.validate(test_data)
        for i in range(self.horizon):
            self._logger.info(f'MAPE {result[i * 3]:7.3f}, MAE {result[i * 3 + 1]:4.3f}, RMSE {result[i * 3 + 2]:6.3f};')

        # with torch.no_grad():
        #     self.model.eval()
        #     preds = []
        #     trues = []
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data):
        #         pred, true = self._process_one_batch(
        #             test_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
        #         preds.append(pred.detach().cpu().numpy())
        #         trues.append(true.detach().cpu().numpy())
        #     preds = np.array(preds)  # [89, 32, pred_len, feature_dim]
        #     trues = np.array(trues)  # [89, 32, pred_len, feature_dim]
        #     print('test shape:', preds.shape, trues.shape)
        #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #     print('test shape:', preds.shape, trues.shape)  # [len, pred_len, feature_dim]
        #     mae, mse, rmse, mape, mspe = metric(preds, trues)
        #     self._logger.info('mse:{}, mae:{}'.format(mse, mae))


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

