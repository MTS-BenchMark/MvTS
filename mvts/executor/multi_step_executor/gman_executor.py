import os
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import copy
import time
from scipy.stats import pearsonr
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model.loss import masked_mae_loss


class GMANExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.patience = self.config.get('patience', 10)
        self.epochs = self.config.get("epochs", 100)
        self.batch_size = self.config.get("batch_size", 64)
        self.validate_freq = self.config.get("validate_freq", 1)
        self.horizon = self.config.get("horizon", 3)
        self.normalize_method = self.config.get("normalize", 0)
        self.evaluator = Evaluator(self.config)

        self.early_stop = self.config.get("early_stop", False)

        self.SE = self.model.SE
        self.scaler = self.model.scaler

        self.optim = self.config.get('optim', "adam")
        self.seq_len = self.config.get('window_size', 12)  # for the encoder
        self.horizon = self.config.get('horizon', 12)  # for the decoder

        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0.001)
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.8)
        self.decay_epoch = self.config.get('decay_epoch', 100)

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
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.learning_rate)
        else:
            self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.my_optim, step_size=self.decay_epoch, gamma=self.lr_decay_ratio)

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


    def _compute_loss(self, y_true, y_predicted):
        y_predicted = self.scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)


    def train(self, train_loader, valid_loader):
        min_val_loss = float('inf')
        self.best_model = copy.deepcopy(self.model)
        wait = 0
        val_loss, val_post = self.validate(valid_loader)
        print(val_post)
        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, (X, Y, TE) in enumerate(pbar):
                self.model.zero_grad()
                X = X.to(self.device)
                Y = Y.to(self.device)
                TE = TE.to(self.device)
                output = self.model(X, TE)
                loss = self._compute_loss(Y, output)
                losses.append(loss.item())
                loss.backward()
                # gradient clipping - this does it in place
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')
            self.my_lr_scheduler.step()
            if (epoch_num+1) % self.validate_freq == 0:
                val_loss, val_post = self.validate(valid_loader)
                self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))
                if val_loss < min_val_loss:
                    wait = 0
                    min_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)
                else:
                    wait += 1
            if wait >= self.patience:
                self._logger.info('early stop!')
                print("early stop!")
                break

    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        self.model = copy.deepcopy(self.best_model)
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            test_loss, test_post = self.validate(test_data)
            self._logger.info('test_loss: {}'.format(test_loss))
            self._logger.info('test_post: {}'.format(test_post))

    def validate(self, dataloader):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(dataloader, ncols=100)
            losses = []
            y_truths = []
            y_preds = []
            for _, (X, Y, TE) in enumerate(pbar):
                X = X.to(self.device)
                Y = Y.to(self.device)
                TE = TE.to(self.device)
                output = self.model(X, TE) #[12, 128, 207]
                loss = self._compute_loss(Y, output)
                losses.append(loss.item())
                y_truths.append(Y.cpu().numpy())
                y_preds.append(output.cpu().numpy())
            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=0) #[num_data, horizon, num_nodes]
            y_preds = self.scaler.inverse_transform(y_preds)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch dimension
            y_preds = y_preds[:, :, :, None]
            y_truths = y_truths[:, :, :, None]
            escore = self.evaluator.evaluate(y_preds, y_truths)
            message = []
            prediction_length = y_preds.shape[1]
            assert prediction_length == 12
            for i in range(prediction_length):
                mae = escore['MAE'][f'horizon-{i}']
                rmse = escore['RMSE'][f'horizon-{i}']
                mape = escore['MAPE'][f'horizon-{i}']
                pcc = escore['PCC'][f'horizon-{i}']
                message.append("MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}， PCC: {:.4f}".format(mae, mape, rmse, pcc))

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
                "horizon 12": message[11]
            }

            return mean_loss, post_fix


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
