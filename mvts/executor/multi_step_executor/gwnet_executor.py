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
from mvts.executor.utils import get_train_loss
# from mvts.model.loss import masked_mae_np, masked_mape_np, masked_rmse_np, masked_mae


class GWNetExecutor(AbstractExecutor):
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
        self.clip = self.config.get("clip", 5)
        self.early_stop = self.config.get("early_stop", False)
        self.output_dim = self.config.get("output_dim")
        self.evaluator = Evaluator(self.config)

        self.scaler = self.model.scaler

        self.train_loss = self.config.get("train_loss", "masked_mae")
        self.loss = get_train_loss(self.train_loss)

        # self.loss = masked_mae
        self.optim = self.config.get('optim', "adam")
        self.lr = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0.0001)

        if self.optim == 'RMSProp':
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.lr,
                                                weight_decay=self.weight_decay)
        else:
            self.my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


    def train(self, train_loader, valid_loader):
        min_val_loss = float('inf')
        self.best_model = self.model
        wait = 0
        val_loss, val_post = self.validate(valid_loader)
        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, (x, y) in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                self.model.zero_grad()
                trainx = x.type(torch.FloatTensor).to(self.device).transpose(1, 3)
                trainy = y.type(torch.FloatTensor).to(self.device).transpose(1, 3)[:, :self.output_dim, :, :]
                # trainx = torch.Tensor(x).to(self.device).transpose(1, 3) #[64, 12, 207, 2]->[64, 2, 207, 12]
                # trainy = torch.Tensor(y).to(self.device).transpose(1, 3)[:,0,:,:] #[64, 12, 207, 2]->[64, 2, 207, 12]->[64, 207, 12]
                trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
                output = self.model(trainx) #trainx[64, 2, 207, 13] output[64, 12, 207, 1]
                output = output.transpose(1, 3)  #[64, 1, 207, 12]
                # trainy = torch.unsqueeze(trainy, dim=1) #[64, 1, 207, 12]
                predict = self.scaler.inverse_transform(output)
                trainy = self.scaler.inverse_transform(trainy)
                loss = self.loss(predict, trainy)
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.my_optim.step()
                losses.append(loss.item())
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')

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
        # preds.shape: [horizon, total_time, nodes_num, feature]
        preds = preds.transpose(1, 0, 2, 3)
        truths = truths.transpose(1, 0, 2, 3)
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
                x = x.type(torch.FloatTensor).to(self.device).transpose(1, 3)
                real = y.type(torch.FloatTensor).to(self.device).transpose(1, 3)[:, :self.output_dim, :, :]
                # x = torch.Tensor(x).to(self.device).transpose(1, 3) #[64, 12, 207, 2]->[64, 2, 207, 12]
                # y = torch.Tensor(y).to(self.device).transpose(1, 3)[:, 0, :, :] #[64, 12, 207, 2]->[64, 2, 207, 12]->[64, 207, 12]

                input = nn.functional.pad(x, (1, 0, 0, 0))
                output = self.model(input) #output[64, 12, 207, 1]
                output = output.transpose(1, 3) #[64, 1, 207, 12]
                # output = [batch_size,12,num_nodes,1]
                # real = torch.unsqueeze(y, dim=1) #[64, 1, 207, 12]
                real = self.scaler.inverse_transform(real)
                predict = self.scaler.inverse_transform(output) #[64, 1, 207, 12]
                loss = self.loss(predict, real)
                losses.append(loss.item())
                # predict = torch.squeeze(predict, dim=1) #[64, 2, 207, 12]
                # real = torch.squeeze(real, dim=1)  # [64, 2, 207, 12]
                predict = predict.permute(3, 0, 2, 1) #[12, 64, 207, 2]
                real = real.permute(3, 0, 2, 1)
                y_truths.append(real.cpu())
                y_preds.append(predict.cpu())

            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            post_fix = self.calculate_metrics(y_preds, y_truths, type='valid')
            print(post_fix)
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
