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


class STGCNExecutor(MultiStepExecutor):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.seq_len = self.config.get('window_size', 12)
        self.output_dim = self.config.get('output_dim', 1)
        self.input_dim = self.config.get('input_dim', 1)
        self.use_curriculum_learning = self.config.get('use_curriculum_learning', False)
        self.horizon = self.config.get('horizon', 1)  # for the decoder
        adj_mx = self.model.adj_mx
        self.A_wave = torch.from_numpy(self._get_normalized_adj(adj_mx)).to(self.device)


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
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y


    def train(self, train_data, valid_data):
        print("begin training")
        wait = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss = []
            train_data.shuffle()

            for iter, (x,y) in enumerate(train_data.get_iterator()):
                self.model.train()
                self.model.zero_grad()
                x = torch.Tensor(x)
                y = torch.Tensor(y)
                trainx, trainy = self._prepare_data(x, y)
                output = self.model(self.A_wave, trainx)
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
                x = torch.Tensor(x)
                y = torch.Tensor(y)
                valx, valy = self._prepare_data(x, y) 
                with torch.no_grad():
                    output = self.model(self.A_wave, valx)
                valy = valy.permute(1, 0, 2)
                output = output.permute(1, 0, 2)
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
        outputs = []
        realy = []
        seq_len = test_data.seq_len  #test_data["y_test"]
        self.model.eval()
        for iter, (x, y) in enumerate(test_data.get_iterator()):
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            testx, testy = self._prepare_data(x, y) 
            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(self.A_wave, testx)
                outputs.append(pred.permute(1, 0, 2)) 
                realy.append(testy.permute(1, 0, 2))
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

    
    def _get_normalized_adj(self, A):
        """
        Returns the degree normalized adjacency matrix.
        """
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                            diag.reshape((1, -1)))
        return A_wave
