import os
from datetime import datetime
import numpy as np
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
from mvts.model.loss import masked_mae_np, masked_mape_np, masked_rmse_np
from mvts.model.loss import masked_mae_loss


class DCRNNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.patience = self.config.get('patience', 10)
        self.epoch_num = self.config.get("epoch", 0)
        self.epochs = self.config.get("epochs", 100)
        self.batch_size = self.config.get("batch_size", 64)
        self.validate_freq = self.config.get("validate_freq", 1)
        self.horizon = self.config.get("horizon", 3)
        self.normalize_method = self.config.get("normalize", 0)
        self.evaluator = Evaluator(self.config)

        self.dataset = self.config.get("dataset", "METR-LA")
        self.early_stop = self.config.get("early_stop", False)

        self.adj_mx = self.model.adj_mx
        self.scaler = self.model.scaler
        self.num_batches = self.model.num_batches

        self.optim = self.config.get('optim', "adam")
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.num_nodes = self.config.get('num_nodes', 1)
        self.input_dim = self.config.get('input_dim', 1)
        self.window = self.config.get('window', 12)  # for the encoder
        self.output_dim = self.config.get('output_dim', 1)
        self.use_curriculum_learning = self.config.get('use_curriculum_learning', False)
        self.horizon = self.config.get('horizon', 1)  # for the decoder

        self.base_lr = self.config.get('base_lr', 0.001)
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.min_learning_rate = self.config.get('min_learning_rate', 2.0e-06)
        self.steps = self.config.get('steps', [20, 30, 40, 50])
        self.epsilon = self.config.get('epsilon', 1.0e-3)

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
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.base_lr, eps=self.epsilon)
        else:
            self.my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.base_lr, eps=self.epsilon)
        self.my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.my_optim, milestones=self.steps, gamma=self.lr_decay_ratio)


        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        #TODO
        x = x.float()
        y = y.float()
        # print('x.type(): ', x.type())
        # exit()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        # x = x.view(self.window, batch_size, self.num_nodes * self.input_dim)
        x = x.reshape(self.window, batch_size, self.num_nodes * self.input_dim)
        # y = y[..., :self.output_dim].view(self.horizon, batch_size, self.num_nodes * self.output_dim)
        y = y[..., :self.output_dim].reshape(self.horizon, batch_size, self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.scaler.inverse_transform(y_true)
        y_predicted = self.scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)


    def train(self, train_loader, valid_loader):
        min_val_loss = float('inf')
        self.best_model = self.model
        wait = 0
        batches_seen = self.num_batches * self.epoch_num

        val_loss, val_post = self.validate(valid_loader)
        self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))

        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, (x, y) in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                # y的最后一维也是input_dim，即2
                self.model.zero_grad()
                x, y = self._prepare_data(x, y) #y.shape:[seq_length, batch_size, nodes_num]
                #x.shape: [window, batch_size, nodes_num*input_dim]
                #y.shape: [horizon, batch_size, nodes_num*output_dim]
                output = self.model(x, y, batches_seen)
                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, eps=self.epsilon)
                batches_seen += 1
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')
            self.my_lr_scheduler.step()
            if (epoch_num+1) % self.validate_freq == 0:
                val_loss, val_post = self.validate(valid_loader)
                self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))
                if val_loss < min_val_loss:
                    wait = 0
                    min_val_loss = val_loss
                    self.best_model = self.model
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
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            test_loss, test_post = self.validate(test_data)
            self._logger.info('test_loss: {}'.format(test_loss))
            self._logger.info('test_post: {}'.format(test_post))

    def calculate_metrics(self, preds, truths, type):
        # preds.shape: [seq_length, total_time, nodes_num*output_dim]
        y_preds_scaled = preds.transpose(1, 0, 2)
        y_truths_scaled = truths.transpose(1, 0, 2)
        seq_len = y_truths_scaled.shape[0]
        y_preds_scaled = y_preds_scaled.reshape(seq_len, self.horizon, self.num_nodes, self.output_dim)
        y_truths_scaled = y_truths_scaled.reshape(seq_len, self.horizon, self.num_nodes, self.output_dim)

        #evaluator要求的数据格式为[total_time, seq_length, nodes_num, num_features]
        escore = self.evaluator.evaluate(y_preds_scaled, y_truths_scaled)
        message = []
        prediction_length = y_preds_scaled.shape[1]
        assert prediction_length == 12
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

    def validate(self, dataloader):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        print('begin validate')
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(dataloader, ncols=100)
            losses = []
            y_truths = []
            y_preds = []
            for _, (x, y) in enumerate(pbar):
                x, y = self._prepare_data(x, y)
                # x.shape: [window, batch_size, nodes_num*input_dim]
                # y.shape: [horizon, batch_size, nodes_num*output_dim]
                output = self.model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())
            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=1) #[horizon, len, nodes_num*output_dim]
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.scaler.inverse_transform(y_truths[t]) #[3425, 207]
                #print('y_truth.shape: ', y_truth.shape) [3425, 207]
                y_pred = self.scaler.inverse_transform(y_preds[t])
                y_truth = torch.from_numpy(y_truth).unsqueeze(0)
                y_pred = torch.from_numpy(y_pred).unsqueeze(0)
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            y_truths_scaled = np.concatenate(y_truths_scaled, axis=0)
            y_preds_scaled = np.concatenate(y_preds_scaled, axis=0)
            post_fix = self.calculate_metrics(y_preds_scaled, y_truths_scaled, type='valid')

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
