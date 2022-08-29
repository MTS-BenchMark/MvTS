import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logging import getLogger
from mvts.evaluator.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.utils import get_evaluator, ensure_dir
from mvts.model import loss
from functools import partial

class Optim(object):
    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def step(self):
        # Compute gradients norm.
        grad_norm = 0

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


class MTGNNExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(self.device)
        self.cuda = self.config.get('cuda', True)
        self.model = model.to(self.device)
        self.save = self.config.get('save', "")
        self.patience = self.config.get('patience', 10)

        self.optim = self.config.get('optim', "adam")
        self.lr = self.config.get('lr', 0.001)
        self.clip = self.config.get('clip', 5)
        self.weight_decay = self.config.get('weight_decay', 0.0001)
        self.evaluator = Evaluator(self.config)

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

        self.L1Loss = self.config.get("L1Loss", True)
        if self.L1Loss:
            self.criterion = nn.L1Loss(size_average=False).to(self.device)
        else:
            self.criterion = nn.MSELoss(size_average=False).to(self.device)
        self.evaluateL2 = nn.MSELoss(size_average=False).to(self.device)
        self.evaluateL1 = nn.L1Loss(size_average=False).to(self.device)

        self.best_val = 10000000
        self.optim = Optim(
            self.model.parameters(), self.optim, self.lr, self.clip,
            lr_decay=self.weight_decay)

        self.epochs = self.config.get("epochs", 100)
        self.scaler = self.model.scaler
        self.scale = self.model.scale
        self.rse = self.model.rse
        self.rae = self.model.rae
        self.num_nodes = self.config.get("num_nodes", 0)
        self.num_split = self.config.get("num_split", 1)
        self.step_size = self.config.get('step_size', 100)
        self.batch_size = self.config.get("batch_size", 64)

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.to(self.device)
                Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    def _evaluate(self, X, Y, model, evaluateL2, evaluateL1, batch_size):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in self.get_batches(X, Y, batch_size, False):
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            with torch.no_grad():
                output = model(X)
            output = torch.squeeze(output)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))

            # scale = self.scaler.expand(output.size(0), self.num_nodes)
            output = self.scaler.inverse_transform(output)
            Y = self.scaler.inverse_transform(Y)
            total_loss += evaluateL2(output, Y).item()
            total_loss_l1 += evaluateL1(output, Y).item()
            n_samples += (output.size(0) * self.num_nodes)

        rse = math.sqrt(total_loss / n_samples) / self.rse
        rae = (total_loss_l1 / n_samples) / self.rae

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        print('rse:{}, rae:{}, corr:{}'.format(rse, rae, correlation))

        predict = self.scaler.inverse_transform(predict)
        Ytest = self.scaler.inverse_transform(Ytest)
        escore = self.evaluator.evaluate(predict, Ytest)
        rse = escore['rse']['all']
        rae = escore['rae']['all']
        correlation = escore["node_pcc"]['all']
        print('rse:{}, rae:{}, corr:{}'.format(rse, rae, correlation))

        scale = self.scale.expand(predict.shape[0], self.num_nodes).data.cpu().numpy()
        escore = self.evaluator.evaluate(predict*scale, Ytest*scale)
        rse = escore['rse']['all']
        rae = escore['rae']['all']
        correlation = escore["node_pcc"]['all']
        print('rse:{}, rae:{}, corr:{}'.format(rse, rae, correlation))

        return rse, rae, correlation, escore

    def _train(self, X, Y, model, criterion, optim, batch_size):
        model.train()
        total_loss = 0
        n_samples = 0
        iter = 0
        for X, Y in self.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            if iter % self.step_size == 0:
                perm = np.random.permutation(range(self.num_nodes))
            num_sub = int(self.num_nodes / self.num_split)

            for j in range(self.num_split):
                if j != self.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(self.device)
                tx = X[:, :, id, :]
                ty = Y[:, id]
                output = model(tx, id)
                output = torch.squeeze(output)
                # scale = self.scaler.expand(output.size(0), self.num_nodes)
                # scale = scale[:, id]
                loss = criterion(output, ty)
                loss.backward()
                total_loss += loss.item()
                n_samples += (output.size(0) * self.num_nodes)
                grad_norm = optim.step()

            if iter % 100 == 0:
                print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * self.num_nodes)))
            iter += 1
        return total_loss / n_samples

    def train(self, train_data, valid_data):
        print('begin training')
        patience = 0
        self.model.train()
        self.best_model = self.model
        val_loss, val_rae, val_corr, _ = self._evaluate(valid_data[0], valid_data[1], self.model, self.evaluateL2,
                                                     self.evaluateL1,
                                                     self.batch_size)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss = self._train(train_data[0], train_data[1], self.model, self.criterion, self.optim, self.batch_size)
            val_loss, val_rae, val_corr, _ = self._evaluate(valid_data[0], valid_data[1], self.model, self.evaluateL2, self.evaluateL1,
                                                   self.batch_size)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < self.best_val:
                patience = 0
                # with open(self.save, 'wb') as f:
                #     torch.save(self.model, f)
                self.best_val = val_loss
                self.best_model = self.model
            else:
                patience += 1

            if patience > self.patience:
                break
            # if epoch % 5 == 0:
            #     test_acc, test_rae, test_corr = self._evaluate(Data.test[0], Data.test[1], model, evaluateL2,
            #                                              evaluateL1,
            #                                              args.batch_size)
            #     print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr),
            #           flush=True)

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
            test_acc, test_rae, test_corr, escore = self._evaluate(test_data[0], test_data[1], self.model, self.evaluateL2,
                                                           self.evaluateL1,
                                                           self.batch_size)
            print(escore)

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






