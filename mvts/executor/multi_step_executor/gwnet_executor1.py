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
from mvts.executor.multi_step_executor.multi_step_executor import MultiStepExecutor
from mvts.executor.utils import get_train_loss
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model import loss
from functools import partial



class GWNETExecutor(MultiStepExecutor):
    def __init__(self, config, model):
        super().__init__(config, model)

    def train(self, train_data, valid_data):
        print("begin training")
        device = self.device
        wait = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss = []
            train_data.shuffle()

            for iter, (x,y) in enumerate(train_data.get_iterator()):
                self.model.train()
                self.model.zero_grad()
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
                trainy = torch.Tensor(y).to(device)
                output = self.model(trainx)
                # print(output.shape)
                # print(trainy.shape)
                # print("*****************************")
                
                loss = self.criterion(self.scaler.inverse_transform(output), \
                    self.scaler.inverse_transform(trainy))
                loss.backward()
                self.optim.step()
                train_loss.append(loss.item())
            
            if self.lr_decay:
                self.optim.lr_scheduler.step()

            valid_loss = []
            valid_mape = []
            valid_rmse = []
            valid_pcc = []
            for iter, (x, y) in enumerate(valid_data.get_iterator()):
                self.model.eval()
                valx = torch.Tensor(x).to(device)
                valx= valx.transpose(1, 3)
                valx = nn.functional.pad(valx, (1, 0, 0, 0))
                valy = torch.Tensor(y).to(device) #[64, 12, 207, 1]
                with torch.no_grad():
                    output = self.model(valx) 
                # print(output.shape)
                # print(valy.shape)
                # print("%%%%%%%%%%%%%%%%%%%%%%%%")
                score = self.evaluator.evaluate(self.scaler.inverse_transform(output), \
                    self.scaler.inverse_transform(valy))
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
        device = self.device
        outputs = []
        realy = []
        seq_len = test_data.seq_len  #test_data["y_test"]
        self.model.eval()
        for iter, (x, y) in enumerate(test_data.get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx= testx.transpose(1, 3)
            testx = nn.functional.pad(testx, (1, 0, 0, 0))
            testy = torch.Tensor(y).to(device) #[64, 12, 207, 1]
            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(testx) #[64, 12, 207, 1]
                outputs.append(pred) 
                realy.append(testy)
        realy = torch.cat(realy, dim=0)
        yhat = torch.cat(outputs, dim=0)

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
