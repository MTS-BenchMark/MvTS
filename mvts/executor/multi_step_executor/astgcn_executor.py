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
# from mvts.model.loss import masked_mae_np, masked_mape_np, masked_rmse_np
# from mvts.evaluator.utils import masked_mae_loss
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error

def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class ASTGCNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.evaluator = Evaluator(self.config)
        self.model = model.to(self.device)
        self.patience = self.config.get('patience', 10)
        self.epochs = self.config.get("epochs", 100)
        self.finetune_epochs = self.config.get("fine_tune_epochs", 50)
        self.batch_size = self.config.get("batch_size", 64)
        self.validate_freq = self.config.get("validate_freq", 1)

        self.early_stop = self.config.get("early_stop", False)

        self.adj_mx = self.model.adj_mx
        self.scaler = self.model.scaler

        self.learning_rate = self.config.get('learning_rate', 0.001)

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

        self.loss = nn.L1Loss().to(self.device)  # 定义损失函数
        self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器，传入所有网络参数

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


    def train(self, train_data, valid_data):
        self.train_data = train_data
        self.valid_data = valid_data
        train_loader, train_target_tensor = train_data[0], train_data[1]
        min_val_loss = float('inf')
        self.best_model = self.model
        wait = 0

        val_loss, val_post, _, _ = self.validate(valid_data)
        self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))

        for epoch_num in range(0, self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, batch_data in enumerate(pbar): #[batch_size, seq_length, nodes_num, input_dim]
                self.model.zero_grad()
                encoder_inputs, decoder_inputs, labels = batch_data
                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
                decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
                labels = labels.unsqueeze(-1)
                outputs = self.model(encoder_inputs, decoder_inputs)
                loss = self.loss(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                self.my_optim.step()
                pbar.set_description(f'train epoch: {epoch_num}, train loss: {np.mean(losses):.8f}')

            if (epoch_num + 1) % self.validate_freq == 0:
                val_loss, val_post, _, _ = self.validate(valid_data)
                self._logger.info('val_loss: {}, val_post: {}'.format(val_loss, val_post))
                if val_loss < min_val_loss:
                    wait = 0
                    min_val_loss = val_loss
                    self.best_model = self.model
                else:
                    wait += 1
            if self.early_stop:
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
            test_loss, test_post, _, _ = self.validate(test_data)
            self._logger.info('test_loss: {}'.format(test_loss))
            self._logger.info('test_post: {}'.format(test_post))

        #finetune
        self._logger.info('Start evaluating ...')
        self.finetune()

        #final evaluate
        self._logger.info('Start final evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            test_loss, test_post, mae_all, rmse_all = self.validate(test_data)
            self._logger.info('test_loss: {}'.format(test_loss))
            self._logger.info('test_post: {}'.format(test_post))
            self._logger.info('rmse_all: {}'.format(rmse_all))
            self._logger.info('mae_all: {}'.format(mae_all))


    def finetune(self):
        train_data = self.train_data
        valid_data = self.valid_data
        train_loader, train_target_tensor = train_data[0], train_data[1]
        min_val_loss = float('inf')
        self.best_model = self.model
        self.new_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate*0.1)  # 定义优化器，传入所有网络参数

        for epoch_num in range(self.epochs, self.epochs + self.finetune_epochs):
            self.model.train()
            pbar = tqdm(train_loader, ncols=100)
            losses = []
            for step, batch_data in enumerate(pbar):  # [batch_size, seq_length, nodes_num, input_dim]
                self.model.zero_grad()
                encoder_inputs, decoder_inputs, labels = batch_data
                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
                decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
                labels = labels.unsqueeze(-1)
                predict_length = labels.shape[2]  # T
                # encode
                encoder_output = self.model.encode(encoder_inputs)
                # decode
                decoder_start_inputs = decoder_inputs[:, :, :1, :]
                decoder_input_list = [decoder_start_inputs]

                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.model.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]

                loss = self.loss(predict_output, labels)  # 计算误差
                losses.append(loss.item())
                loss.backward()
                self.new_optim.step()
                pbar.set_description(f'finetune train epoch: {epoch_num}, finetune train loss: {np.mean(losses):.8f}')

            if (epoch_num + 1) % self.validate_freq == 0:
                val_loss, val_post, _, _ = self.validate(valid_data)
                self._logger.info('finetune val_loss: {}, finetune val_post: {}'.format(val_loss, val_post))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    self.best_model = self.model


    def validate(self, data):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model.eval()
            data_loader, data_target_tensor = data[0], data[1]
            pbar = tqdm(data_loader, ncols=100)
            losses = []
            # inputs = []
            y_truths = []
            y_preds = []
            for _, batch_data in enumerate(pbar):
                encoder_inputs, decoder_inputs, labels = batch_data
                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
                decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
                labels = labels.unsqueeze(-1)  # (B，N，T，1)
                predict_length = labels.shape[2]  # T

                # encode
                encoder_output = self.model.encode(encoder_inputs)
                # print('encoder_output:', encoder_output.shape)
                # decode
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                decoder_input_list = [decoder_start_inputs]
                # 按着时间步进行预测
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.model.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]

                loss = self.loss(predict_output, labels)  # 计算误差
                losses.append(loss.item())
                # y_truths.append(labels.cpu())
                # y_preds.append(predict_output.cpu())
                # inputs.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)
                y_truths.append(labels.cpu().numpy())
                y_preds.append(predict_output.detach().cpu().numpy())

            mean_loss = np.mean(losses)


            y_preds = np.concatenate(y_preds, 0)  # (batch, N, T', 1)
            y_preds = self.scaler.inverse_transform(y_preds)
            data_target_tensor = self.scaler.inverse_transform(data_target_tensor.detach().cpu().numpy())

            print('prediction:', y_preds.shape)  # [N, num_nodes, horizon, 1]
            print('data_target_tensor:', data_target_tensor.shape)  # [N, num_nodes, horizon]

            prediction_length = y_preds.shape[2]

            # escore = self.evaluator.evaluate(y_preds.permute(0, 2, 1)[:, :, :, 0], data_target_tensor.permute(0, 2, 1)[:, :, :])
            escore = self.evaluator.evaluate(np.transpose(y_preds, (0, 2, 1, 3))[:, :, :, 0],
                                             np.transpose(data_target_tensor, (0, 2, 1))[:, :, :])

            message = []
            for i in range(prediction_length):
                mae = escore['MAE'][f'horizon-{i}']
                rmse = escore['RMSE'][f'horizon-{i}']
                mape = escore['masked_MAPE'][f'horizon-{i}']
                pcc = escore['PCC'][f'horizon-{i}']
                message.append("MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}， PCC: {:.4f}".format(mae, mape, rmse, pcc))

            mae_all = escore['MAE']['all']
            rmse_all = escore['RMSE']['all']

            print(mae_all, rmse_all)

            post_fix2 = {
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

            return mean_loss, post_fix2, mae_all, rmse_all


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
