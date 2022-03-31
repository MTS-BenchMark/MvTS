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
from libcity.executor.multi_step_executor.multi_step_executor import MultiStepExecutor
from libcity.executor.utils import get_train_loss
from libcity.utils import get_evaluator, ensure_dir, Optim
from libcity.model import loss
from functools import partial


class DGCRNExecutor(MultiStepExecutor):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.seq_len = self.config.get('window_size', 12)
        self.output_dim = self.config.get('output_dim', 1)
        self.input_dim = self.config.get('input_dim', 1)
        self.cl = self.config.get('cl', True)
        self.horizon = self.config.get('horizon', 1)  # for the decoder
        self.step = self.config.get("step_size1", 2500)
        self.seq_out_len = self.config.get("horizon", 12)


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

            for iter, (x,y, ycl) in enumerate(train_data.get_iterator()):
                
                self.model.train()
                self.model.zero_grad()
                
                batches_seen += 1
                trainx = torch.Tensor(x).to(device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)

                trainycl = torch.Tensor(ycl).to(device)
                trainycl = trainycl.transpose(1, 3)


                if self.iter % self.step == 0 and self.task_level < self.seq_out_len:
                    self.task_level += 1
                
                if self.cl:
                    output = self.model(trainx, idx = None, ycl = trainycl, 
                        batches_seen = batches_seen, task_level=self.task_level)
                else:
                    output = self.model(trainx, idx = None, ycl = trainycl, 
                        batches_seen = batches_seen, task_level=self.seq_out_len)
                
                output = output.transpose(1, 3)
                predict = self.scaler.inverse_transform(output)
                
                if self.cl:
                    loss = self.criterion(predict[:, :, :, :self.task_level],
                         trainy[:, :self.output_dim, :, :self.task_level])
                else:
                    loss = self.criterion(predict, trainy[:, :self.output_dim, :, :])

                loss.backward()
                self.optim.step()
                train_loss.append(loss.item())
            
            if self.lr_decay:
                self.optim.lr_scheduler.step()

            valid_loss = []

            for iter, (x, y) in enumerate(valid_data.get_iterator()):
                self.model.eval()

                valx = torch.Tensor(x).to(device)
                valx = valx.transpose(1, 3)
                valy = torch.Tensor(y).to(device)
                valy = valy.transpose(1, 3)

                with torch.no_grad():
                    output = self.model(valx, ycl = valy)

                # output = output.transpose(1, 3)
                predict = self.scaler.inverse_transform(output)
                
                score = self.evaluator.evaluate(predict, valy[:, :self.output_dim, :, :].transpose(1, 3))

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
        outputs = []
        realy = []
        seq_len = test_data.seq_len  #test_data["y_test"]
        device = self.device
        self.model.eval()
        for iter, (x, y) in enumerate(test_data.get_iterator()):

            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(testx, ycl=testy)
                # pred = pred.transpose(1, 3)
            outputs.append(pred) 
            realy.append(testy)

        realy = torch.cat(realy, dim=0)
        yhat = torch.cat(outputs, dim=0)

        realy = realy[:seq_len, ...]
        yhat = yhat[:seq_len, ...]

        preds = self.scaler.inverse_transform(yhat)

        res_scores = self.evaluator.evaluate(preds, realy[:, :self.output_dim, :, :].transpose(1, 3))
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
