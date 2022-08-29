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
# from mvts.model.loss import masked_mae_np, masked_mape_np, masked_rmse_np
# from mvts.evaluator.utils import masked_mae_loss
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error

def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i+1, num_domain+1):
            index.append((i, j))
    return index



class AdaRNNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.device = torch.device(self.device)
        # self.evaluator = Evaluator(self.config, "multi_step")
        self.model = model.to(self.device)

        self.epochs = self.config.get('epochs')
        self.validate_freq = self.config.get('validate_freq')
        self.early_stop = self.config.get('early_stop')
        self.patience = self.config.get('patience')
        self.learning_rate = self.config.get('learning_rate')
        self.num_layers = self.config.get("num_layers")
        self.len_seq = self.config.get("len_seq")
        self.pre_epoch = self.config.get("pre_epoch")
        self.len_win = self.config.get("len_win")
        self.dw = self.config.get("dw")
        self.model_name = self.config.get("model_name")
        self.evaluator = Evaluator(self.config)
        self.saveflag = False

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

        # self.loss = nn.MSELoss().to(self.device)  # 定义损失函数
        print("init model...")
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        self.my_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器，传入所有网络参数

    def transform_type(self, init_weight):
        weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight

    def train_epoch_transfer_Boosting(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
        self.model.train()
        criterion = nn.MSELoss()
        criterion_1 = nn.L1Loss()
        loss_all = []
        loss_1_all = []
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        len_loader = np.inf
        for loader in train_loader_list:
            if len(loader) < len_loader:
                len_loader = len(loader)
        for data_all in tqdm(zip(*train_loader_list), total=len_loader):
            self.my_optim.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                feature, label, label_reg = data[0].to(self.device).float(
                ), data[1].to(self.device).long(), data[2].to(self.device).float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = get_index(len(data_all) - 1)
            for temp_index in index:
                s1 = temp_index[0]
                s2 = temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).to(self.device)
            for i in range(len(index)):
                feature_s = list_feat[index[i][0]]
                feature_t = list_feat[index[i][1]]
                label_reg_s = list_label[index[i][0]]
                label_reg_t = list_label[index[i][1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(
                    feature_all, weight_mat)
                dist_mat = dist_mat + dist
                pred_s = pred_all[0:feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0):]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)
                loss_l1 = criterion_1(pred_s, label_reg_s)

                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer

            loss_all.append(
                [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
            loss_1_all.append(loss_l1.item())
            self.my_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.)
            self.my_optim.step()
        loss = np.array(loss_all).mean(axis=0)
        loss_l1 = np.array(loss_1_all).mean()
        if epoch > 0:  # args.pre_epoch:
            weight_mat = self.model.update_weight_Boosting(
                weight_mat, dist_old, dist_mat)
        return loss, loss_l1, weight_mat, dist_mat

    def train_AdaRNN(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
        self.model.train()
        criterion = nn.MSELoss()
        criterion_1 = nn.L1Loss()
        loss_all = []
        loss_1_all = []
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        len_loader = np.inf
        for loader in train_loader_list:
            if len(loader) < len_loader:
                len_loader = len(loader)
        for data_all in tqdm(zip(*train_loader_list), total=len_loader):
            self.my_optim.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                feature, label, label_reg = data[0].to(self.device).float(
                ), data[1].to(self.device).long(), data[2].to(self.device).float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = get_index(len(data_all) - 1)
            for temp_index in index:
                s1 = temp_index[0]
                s2 = temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).to(self.device)
            for i in range(len(index)):
                feature_s = list_feat[index[i][0]]
                feature_t = list_feat[index[i][1]]
                label_reg_s = list_label[index[i][0]]
                label_reg_t = list_label[index[i][1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                if epoch < self.pre_epoch:
                    pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
                        feature_all, len_win=self.len_win)
                else:
                    pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(
                        feature_all, weight_mat)
                    dist_mat = dist_mat + dist
                pred_s = pred_all[0:feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0):]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)
                loss_l1 = criterion_1(pred_s, label_reg_s)

                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer
            loss_all.append(
                [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
            loss_1_all.append(loss_l1.item())
            self.my_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.)
            self.my_optim.step()
        loss = np.array(loss_all).mean(axis=0)
        loss_l1 = np.array(loss_1_all).mean()
        if epoch >= self.pre_epoch:
            if epoch > self.pre_epoch:
                weight_mat = self.model.update_weight_Boosting(
                    weight_mat, dist_old, dist_mat)
            return loss, loss_l1, weight_mat, dist_mat
        else:
            weight_mat = self.transform_type(out_weight_list)
            return loss, loss_l1, weight_mat, None

    def train(self, train_data, valid_data):
        val_loss, val_loss_l1, val_loss_r = self.validate(valid_data)
        min_val_loss = float('inf')
        wait = 0
        time_now = time.time()
        weight_mat, dist_mat = None, None

        for epoch_num in range(0, self.epochs):
            self._logger.info('Epoch:{}'.format(epoch_num))
            self._logger.info('training...')
            if self.model_name in ['Boosting']:
                loss, loss1, weight_mat, dist_mat = self.train_epoch_transfer_Boosting(train_data, epoch_num, dist_mat, weight_mat)
            elif self.model_name in ['AdaRNN']:
                loss, loss1, weight_mat, dist_mat = self.train_AdaRNN(train_data, epoch_num, dist_mat, weight_mat)
            else:
                self._logger.info("error in model_name!")
            # self._logger.info(loss, loss1)

            if epoch_num % self.validate_freq == 0:
                val_loss, val_loss_l1, val_loss_r = self.validate(valid_data)
                self._logger.info('Epoch: %d, valid_loss_l1: %.6f' % (epoch_num, val_loss_l1))
                # self._logger.info(f'Epoch {epoch_num}, valid_loss_l1: {val_loss_l1}')

                if val_loss < min_val_loss:
                    wait = 0
                    min_val_loss = val_loss
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
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            total_loss_1 = 0
            total_loss_r = 0

            for feature, label, label_reg in tqdm(data, total=len(data)):
                feature, label_reg = feature.to(self.device).float(), label_reg.to(self.device).float()
                with torch.no_grad():
                    pred = self.model.predict(feature)
                # preds.append(pred.cpu().numpy())
                # truths.append(label_reg.cpu().numpy())
                escore = self.evaluator.evaluate(pred, label_reg)
                loss = (escore['RMSE']['all']) ** 2
                loss_r = escore['RMSE']['all']
                loss_1 = escore['MAE']['all']
                total_loss += loss
                total_loss_1 += loss_1
                total_loss_r += loss_r

            loss = total_loss / len(data)
            loss_1 = total_loss_1 / len(data)
            loss_r = total_loss_r / len(data)

            self.model.train()
            # preds = np.concatenate(preds, axis=0)
            # truths = np.concatenate(truths, axis=0)
            # escore = self.evaluator.evaluate(preds, truths)
            # mse = (escore['RMSE']['all']) ** 2
            # rmse = escore['RMSE']['all']
            # mae = escore['MAE']['all']

            return loss, loss_1, loss_r


    def evaluate(self, test_data):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        if self.saveflag:
            self.load_model(self.cache_dir)

        self.model.eval()
        total_loss = 0
        total_loss_1 = 0
        total_loss_r = 0

        for feature, label, label_reg in tqdm(test_data, total=len(test_data)):
            feature, label_reg = feature.to(self.device).float(), label_reg.to(self.device).float()
            with torch.no_grad():
                pred = self.model.predict(feature)

            escore = self.evaluator.evaluate(pred, label_reg)
            loss = (escore['RMSE']['all']) ** 2
            loss_r = escore['RMSE']['all']
            loss_1 = escore['MAE']['all']
            total_loss += loss
            total_loss_1 += loss_1
            total_loss_r += loss_r

        loss = total_loss / len(test_data)
        loss_1 = total_loss_1 / len(test_data)
        loss_r = total_loss_r / len(test_data)


        self._logger.info('MSE: %.6f, L1: %.6f, RMSE: %.6f' % (loss, loss_1, loss_r))
        self._logger.info('Finished.')


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

