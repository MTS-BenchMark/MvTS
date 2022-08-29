import numpy as np
import torch
import math
import time
import torch.nn as nn
from mvts.executor.multi_step_executor.multi_step_executor import MultiStepExecutor
from mvts.utils import get_evaluator, ensure_dir, Optim

class AGCRNExecutor(MultiStepExecutor):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.seq_len = self.config.get('window_size', 12)
        self.output_dim = self.config.get('output_dim', 1)
        self.input_dim = self.config.get('input_dim', 2)
        self.use_curriculum_learning = self.config.get('use_curriculum_learning', False)
        self.horizon = self.config.get('horizon', 1)  # for the decoder
        self.teacher_forcing = self.config.get('teacher_forcing', False)
        self.tf_decay_steps = self.config.get('tf_decay_steps', 2000)
        self.real_value = self.config.get('real_value', True)
        self.grad_norm = self.config.get('grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 5)


    def train(self, train_data, valid_data):
        print("begin training")
        device = self.device
        wait = 0
        batches_seen = self.num_batches * 0
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        self.train_per_epoch = 1


        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss = []
            train_data.shuffle()

            for iter, (x, y) in enumerate(train_data.get_iterator()):
                self.model.train()
                self.model.zero_grad()

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y).to(device)

                trainx = x[..., :self.input_dim]
                trainy = y[..., :self.output_dim]

                if self.teacher_forcing:
                    global_step = (epoch - 1) * self.train_per_epoch + iter
                    teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.tf_decay_steps)
                else:
                    teacher_forcing_ratio = 1.

                # data and target shape: B, T, N, F; output shape: B, T, N, F
                output = self.model(trainx, trainy, teacher_forcing_ratio=teacher_forcing_ratio)

                if self.real_value:
                    trainy = self.scaler.inverse_transform(trainy)
                loss = self.criterion(output, trainy)

                loss.backward()
                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim.step()
                train_loss.append(loss.item())


            if self.lr_decay:
                self.optim.lr_scheduler.step()

            valid_loss = []

            for iter, (x, y) in enumerate(valid_data.get_iterator()):
                self.model.eval()

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y).to(device)

                valx = x[..., :self.input_dim]
                valy = y[..., :self.output_dim]
                with torch.no_grad():
                    output = self.model(valx)
                if self.real_value:
                    valy = self.scaler.inverse_transform(valy)
                score = self.evaluator.evaluate(output, valy)
                
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
        self.model.eval()
        for iter, (x, y) in enumerate(test_data.get_iterator()):
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)

            testx = x[..., :self.input_dim]
            testy = y[..., :self.output_dim] #[64, 12, 207, 1]

            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(testx, teacher_forcing_ratio=0) #[64, 12, 207, 1]
                outputs.append(pred)
                realy.append(testy)

        if self.real_value:
            preds = torch.cat(outputs, dim=0)
        else:
            preds = self.scaler.inverse_transform(torch.cat(outputs, dim=0))

        realy = torch.cat(realy, dim=0)
        #preds = torch.cat(outputs, dim=0)

        realy = self.scaler.inverse_transform(realy)

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

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))