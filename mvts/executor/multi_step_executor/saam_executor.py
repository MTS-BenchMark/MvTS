import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
import tqdm
from torch.utils.tensorboard import SummaryWriter
from mvts.executor.abstract_executor import AbstractExecutor
from mvts.executor.utils import get_train_loss
from mvts.utils import get_evaluator, ensure_dir, Optim
from mvts.model import loss
from functools import partial
from torch.distributions.normal import Normal

class GaussianLoss(nn.Module):
    def __init__(self,mu,sigma):
        """Compute the negative log likelihood of Gaussian Distribution"""
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma
    def forward(self,x):
        loss = - Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss)/(loss.size(0)*loss.size(1))


class SAAMExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.config = config
        self.evaluator = get_evaluator(config)

        _device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(_device)
        
        #self.model = nn.DataParallel(model).to(self.device)
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
        
        self.num_nodes = self.config.get("num_nodes", 0)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 20)
        self.lr_decay = self.config.get("lr_decay", False)
        self.acc_steps = self.config.get("acc_steps")
        self.enc_len = self.config.get("enc_len")
        self.dec_len = self.config.get("dec_len")
        self.pred_samples = self.config.get("pred_samples")


    def valid(self, train_seq,gt,series_id,scaling_factor, encoder):

        with torch.no_grad():
            mu, sigma, h, h_filtered = encoder(series_id,train_seq)
            scaling_factor = torch.unsqueeze(torch.unsqueeze(scaling_factor,dim=1),dim=1)
            criterion = GaussianLoss(scaling_factor*mu,scaling_factor*sigma)
            gt = torch.unsqueeze(gt,dim=2)
            loss = criterion(gt)

        return loss.item(), h, h_filtered


    def validIters(self, valid_loader):
        self.model.eval()
        total_loss = 0
        for id, data in enumerate(valid_loader):
            train_seq,gt,series_id, scaling_factor = data  # (batch_size,seq, value) --> (seq,batch_size,value)
            train_seq = train_seq.to(torch.float32).to(self.device)
            gt = gt.to(torch.float32).to(self.device)
            series_id = series_id.to(self.device)
            scaling_factor = scaling_factor.to(torch.float32).to(self.device)
            loss, h, h_filtered = self.valid(train_seq, gt, series_id, scaling_factor, self.model)
            total_loss += loss
        loss = total_loss/(id+1)
        return loss, h, h_filtered #printeo el ultimo h_filtered que es el que se quedará guardado


    def evaluate_pred(self, train_seq, enc_len, dec_len, series_id, scaling_factor, encoder, pred_samples):
        with torch.no_grad():
            batch_size = train_seq.size(0)
            Pred_series = torch.zeros(pred_samples,dec_len,batch_size,1,device=self.device)
            for i in range(dec_len):
                mu, sigma, _, _ = encoder(series_id,train_seq[:,:(enc_len+1+i),:])
                mu = mu[:,-1,:]
                sigma = sigma[:,-1,:]
                Gaussian = torch.distributions.normal.Normal(mu,sigma)
                pred = Gaussian.sample(torch.Size([pred_samples]))
                if(i< (dec_len-1)):
                    train_seq[:,enc_len+i+1,0] = torch.squeeze(mu)
                pred = pred*torch.unsqueeze(torch.unsqueeze(scaling_factor,dim=1),dim=0)
                Pred_series[:,i,:,:] = pred
        return torch.squeeze(Pred_series)

    def evaluateIters(self, test_loader, enc_len, dec_len, encoder, pred_samples):
        predictions = []
        encoder.eval()
        for id, data in enumerate(test_loader):
            train_seq,gt,series_id, scaling_factor = data  # (batch_size,seq, value) --> (seq,batch_size,value)
            train_seq = train_seq.to(torch.float32).to(self.device)
            gt = gt.to(torch.float32).to(self.device)
            series_id = series_id.to(self.device)
            scaling_factor = scaling_factor.to(torch.float32).to(self.device)
            pred = self.evaluate_pred(train_seq, enc_len, dec_len, series_id, scaling_factor, encoder, pred_samples)
            predictions.append(pred)

        return predictions


    def train(self, train_data, valid_data):
        print("begin training")
        wait = 0
        tr_loss = 0
        iter_th = 0
        global_step = 0
        loss_best = float('inf')
        best_epoch = -1
        best_model = None

        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            epoch_start_time = time.time()
            train_loss = []

            for step, data in enumerate(train_data):
                train_seq, gt, series_id, scaling_factor = data
                train_seq = train_seq.to(torch.float32).to(self.device)
                gt = gt.to(torch.float32).to(self.device)
                series_id = series_id.to(self.device)
                scaling_factor = scaling_factor.to(torch.float32).to(self.device)

                mu, sigma, h, h_filtered = self.model(series_id, train_seq)
                scaling_factor = torch.unsqueeze(torch.unsqueeze(scaling_factor,dim=1),dim=1)

                criterion = GaussianLoss(scaling_factor * mu, scaling_factor * sigma)

                gt = torch.unsqueeze(gt,dim=2)
                loss = criterion(gt)
                loss = loss / self.acc_steps
                loss.backward()
                tr_loss += loss.item()
                iter_th += 1

                if  (step+1) % self.acc_steps ==0:
                    
                    self.optim.step()
                    self.model.zero_grad()
                    global_step += 1
                    tr_loss = 0
            v_loss, h, h_filtered = self.validIters(valid_data)

            if(v_loss < loss_best):
                loss_best = v_loss
                loss_is_best = True
                best_epoch = epoch
                best_model = self.model
            else:
                loss_is_best = False
                wait += 1

            if wait > self.patience:
                print('early stop at epoch: {:04d}'.format(epoch))
                break
        
        self.model = best_model


    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')

        prediction = self.evaluateIters(test_data, self.enc_len, self.dec_len, self.model, self.pred_samples) #200, 24, 32
        print("!!!!!!!!!!!!!!!!!!!!!")
        print(len(prediction))
        print(prediction[0])
        print(prediction[0].shape)
        for idx, data in enumerate(test_data):
            _,gt,_, _ = data
            print(gt.shape)
            exit(0)
        
        

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
