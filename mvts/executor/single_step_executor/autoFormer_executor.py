import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.executor.utils import get_train_loss
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model import loss
from functools import partial


class AutoFormerExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.config = config
        self.evaluator = get_evaluator(config)

        _device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(_device)
        self.model = model.to(self.device)

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

        self.train_loss = self.config.get("train_loss", "masked_mae")
        self.criterion = get_train_loss(self.train_loss) 

        self.cuda = self.config.get("cuda", True)
        self.best_val = 10000000
        self.optim = Optim.Optim(
            model.parameters(), self.config
        )
        self.epochs = self.config.get("epochs", 100)
        self.scaler = self.model.scaler
        self.num_batches = self.model.num_batches
        self.num_nodes = self.config.get("num_nodes", 0)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 20)
        self.lr_decay = self.config.get("lr_decay", False)
        self.mask = self.config.get("mask", True)
        self.output_attention = self.config.get("output_attention", False)\
        
        self.label_len = self.config.get("label_len", 48)
        # self.pred_len = self.config.get("horizon", 96)
        self.pred_len = 1

        self.input_dim = self.config.get("input_dim")
        self.output_dim = self.config.get("output_dim")
        self.dim = min(self.input_dim, self.output_dim)

        self._model_parameters_init()

    def _model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, train_data, valid_data):
        print("begin training")
        wait = 0
        batches_seen = 0
        device = self.device
        self.iter = 0
        self.task_level = 1

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss = []
            train_data.shuffle()

            for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data.get_iterator()):
                
                self.model.train()
                self.model.zero_grad()
                batches_seen += 1
                
                batch_x = batch_x[..., :self.dim]
                batch_y = batch_y[..., :self.dim]
                batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], -1)
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], -1)

                batch_x = torch.Tensor(batch_x).float().to(self.device)
                batch_y = torch.Tensor(batch_y).float().to(self.device)
                batch_x_mark = torch.Tensor(batch_x_mark).float().to(self.device)
                batch_y_mark = torch.Tensor(batch_y_mark).float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                
                outputs = outputs[:, -self.pred_len:, :]
                batch_y = batch_y[:, -self.pred_len:, :]

                loss = self.criterion(self.scaler.inverse_transform(outputs), 
                    self.scaler.inverse_transform(batch_y))

                loss.backward()
                self.optim.step()
                train_loss.append(loss.item())
            
            if self.lr_decay:
                self.optim.lr_scheduler.step()

            valid_loss = []

            for iter, (x, y, x_mark, y_mark) in enumerate(valid_data.get_iterator()):
                self.model.eval()

                x = x[..., :self.dim]
                y = y[..., :self.dim]

                x = x.reshape(x.shape[0], x.shape[1], -1)
                y = y.reshape(y.shape[0], y.shape[1], -1)

                val_x = torch.Tensor(x).float().to(self.device)
                val_y = torch.Tensor(y).float().to(self.device)
                val_x_mark = torch.Tensor(x_mark).float().to(self.device)
                val_y_mark = torch.Tensor(y_mark).float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(val_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([val_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                with torch.no_grad():
                    if self.output_attention:
                        outputs = self.model(val_x, val_x_mark, dec_inp, val_y_mark)[0]
                    else:
                        outputs = self.model(val_x, val_x_mark, dec_inp, val_y_mark)

                outputs = outputs[:, -self.pred_len:, :]
                val_y = val_y[:, -self.pred_len:, :]


                loss = self.criterion(self.scaler.inverse_transform(outputs), 
                    self.scaler.inverse_transform(val_y))
    
                # output = output.transpose(1, 3)
                predict = self.scaler.inverse_transform(outputs)
                
                score = self.evaluator.evaluate(predict, self.scaler.inverse_transform(val_y))

                if self.mask:
                    vloss = score["masked_MAE"]["all"]
                else:
                    vloss = score["MAE"]["all"]
                    
                valid_loss.append(vloss)
            

            mtrain_loss = np.mean(train_loss)

            mvalid_loss = np.mean(valid_loss)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid mae {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), mtrain_loss, \
                        mvalid_loss))

            if mvalid_loss < self.best_val:
                self.best_val = mvalid_loss
                wait = 0
                self.best_val = mvalid_loss
                self.best_model = self.model
            else:
                wait += 1

            if wait >= self.patience:
                print('early stop at epoch: {:04d}'.format(epoch))
                break
        
        self.model = self.best_model


    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        preds = []
        realy = []
        seq_len = test_data.seq_len  #test_data["y_test"]
        device = self.device
        self.model.eval()
        for iter, (x, y, x_mark, y_mark) in enumerate(test_data.get_iterator()):
            self.model.eval()

            x = x[..., :self.dim]
            y = y[..., :self.dim]
            x = x.reshape(x.shape[0], x.shape[1], -1)
            y = y.reshape(y.shape[0], y.shape[1], -1)

            test_x = torch.Tensor(x).float().to(self.device)
            test_y = torch.Tensor(y).float().to(self.device)
            test_x_mark = torch.Tensor(x_mark).float().to(self.device)
            test_y_mark = torch.Tensor(y_mark).float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(test_y[:, -self.pred_len:, :]).float()
            dec_inp = torch.cat([test_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

            with torch.no_grad():
                if self.output_attention:
                    outputs = self.model(test_x, test_x_mark, dec_inp, test_y_mark)[0]
                else:
                    outputs = self.model(test_x, test_x_mark, dec_inp, test_y_mark)
            
            preds.append(outputs[:, -self.pred_len:, :])
            realy.append(test_y[:, -self.pred_len:, :])

        realy = torch.cat(realy, dim=0)
        yhat = torch.cat(preds, dim=0)

        realy = realy[:seq_len, ...]
        yhat = yhat[:seq_len, ...]

        realy = self.scaler.inverse_transform(realy)
        preds = self.scaler.inverse_transform(yhat)

        res_scores = self.evaluator.evaluate(preds, realy)
        for _index in res_scores.keys():
            print(_index, " :")
            step_dict = res_scores[_index]
            for j, k in step_dict.items():
                print(j, " : ", k.item())
        
        

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
