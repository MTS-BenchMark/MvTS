import os
from datetime import datetime
import numpy as np
import torch
import copy
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
from mvts.model.loss import masked_mae_loss


class FCGAGAExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.output_dim = self.config.get("output_dim")
        self.input_dim = self.config.get("input_dim")
        self.patience = self.config.get('patience', 10)
        self.epochs = self.config.get("epochs", 100)
        self.decay_steps = self.config.get("decay_steps", 3)
        self.batch_size = self.config.get("batch_size", 64)
        self.validate_freq = self.config.get("validate_freq", 1)
        self.seq_len = self.config.get('window_size', 12)
        self.horizon = self.config.get("horizon", 12)
        self.evaluator = Evaluator(self.config)

        self.dataset = self.config.get("dataset", "METR-LA")
        self.early_stop = self.config.get("early_stop", False)

        self.node_id = self.model.node_id
        self.scaler = self.model.scaler

        self.base_lr = self.config.get('base_lr', 0.001)
        self.lr_decay_ratio = self.config.get('decay_rate', 0.5)
        self.epsilon = self.config.get('epsilon', 1.0e-3)

        boundary_step = self.epochs // 10
        boundary_start = self.epochs - boundary_step * self.decay_steps - 1
        # self.steps = list(range(boundary_start, self.epochs, boundary_step))
        self.steps = [1, 2]

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
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' + str(param.requires_grad))

        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.base_lr)
        self.my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.my_optim, milestones=self.steps, gamma=self.lr_decay_ratio)


    def _compute_loss(self, y_true, y_predicted):
        y_true = self.scaler.inverse_transform(y_true)
        y_predicted = self.scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)


    def train(self, train_loader, valid_loader):
        min_val_loss = float('inf')
        self.best_model = copy.deepcopy(self.model)
        wait = 0
        valid_loader, test_loader = valid_loader[0], valid_loader[1]
        val_loss, val_post = self.validate(valid_loader)
        self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))
        test_loss, test_post = self.validate(test_loader)
        self._logger.info('test_loss: {}, test_post: {}'.format(test_loss, test_post))
        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, (x, y) in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                x = x.to(self.device)
                y = y.to(self.device)
                self.model.zero_grad()
                y = y[..., :-1]
                history = x[..., :-1]
                seq_len, horizon, num_nodes, feature_dim = history.shape
                y = y.reshape(seq_len, horizon, num_nodes * feature_dim)
                history = history.reshape(seq_len, horizon, num_nodes * feature_dim)
                time_of_day = x[..., -1]
                time_of_day = time_of_day.repeat_interleave(feature_dim, dim=-1)
                node_id = self.node_id.to(self.device)
                output = self.model(history, time_of_day, node_id)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                loss.backward()
                self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')
            self.my_lr_scheduler.step()
            if (epoch_num+1) % self.validate_freq == 0:
                val_loss, val_post = self.validate(valid_loader)
                self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))
                test_loss, test_post = self.validate(test_loader)
                self._logger.info('test_loss: {}, test_post: {}'.format(test_loss, test_post))
                if val_loss < min_val_loss:
                    wait = 0
                    min_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)
                else:
                    wait += 1
            if wait >= self.patience and self.early_stop:
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
        self.load_model(self.best_model_address)
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
            for _, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                y = y.to(self.device)
                #x/y.shape:[batch_size, horizon, nodes_num, feature_dim+1]
                y = y[..., :-1]
                history = x[..., :-1]
                seq_len, horizon, num_nodes, feature_dim = history.shape
                y = y.reshape(seq_len, horizon, num_nodes*feature_dim)
                history = history.reshape(seq_len, horizon, num_nodes*feature_dim)
                time_of_day = x[..., -1]
                time_of_day = time_of_day.repeat_interleave(feature_dim, dim=-1)
                node_id = self.node_id.to(self.device)
                output = self.model(history, time_of_day, node_id)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())
            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=0) #[len, horizon, num_nodes]
            y_truths = np.concatenate(y_truths, axis=0) #[len, horizon, num_nodes]

            y_preds = self.scaler.inverse_transform(y_preds)
            y_truths = self.scaler.inverse_transform(y_truths)
            seq_len, horizon, dim = y_preds.shape
            y_preds = y_preds.reshape(seq_len, horizon, num_nodes, feature_dim)
            y_truths = y_truths.reshape(seq_len, horizon, num_nodes, feature_dim)

            post_fix = self.calculate_metrics(y_preds, y_truths, type='valid')

            return mean_loss, post_fix

    def calculate_metrics(self, preds, truths, type):
        # preds.shape: [total_time, seq_length, nodes_num]
        escore = self.evaluator.evaluate(preds, truths)
        message = []
        prediction_length = preds.shape[1]
        for i in range(prediction_length):
            mae = escore['masked_MAE'][f'horizon-{i}']
            rmse = escore['masked_RMSE'][f'horizon-{i}']
            mape = escore['masked_MAPE'][f'horizon-{i}']
            message.append("MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(mae, mape, rmse))
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

        return post_fix

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件
        Args:
            cache_name(str): 保存的文件名
        """
        self.best_model_address = cache_name
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
