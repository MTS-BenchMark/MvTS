from mvts.evaluator.abstract_evaluator import AbstractEvaluator

import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from collections import defaultdict


def rse_np(preds, labels):
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mse = np.sum(np.square(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.square(np.subtract(labels, means)).astype('float32'))
    return np.sqrt(mse/labels_mse)


def mae_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
    else:
        mae = np.abs(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32')
    return np.mean(mae)


def rmse_np(preds, labels):
    mse = mse_np(preds, labels)
    return np.sqrt(mse)

def mse_np(preds, labels):
    if isinstance(preds, np.ndarray):
        return np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
    else:
        return np.mean(np.square(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32'))

def mape_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    else:
        mape = np.abs(np.divide(np.subtract(preds.cpu().numpy(), labels.cpu().numpy()).astype('float32'), labels.cpu().numpy()))
    return np.mean(mape)

def rae_np(preds, labels):
    mse = np.sum(np.abs(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.abs(np.subtract(labels, means)).astype('float32'))
    return mse/labels_mse

def nrmse(preds, labels):
    """ Normalized RMSE"""
    if not isinstance(preds, np.ndarray):
        preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    t1 = np.linalg.norm(preds - labels)**2 / np.size(labels)
    t2 = np.sum(abs(labels)) / np.size(labels)
    return np.sqrt(t1) / t2

def nd(preds, labels):
    """ Normalized deviation"""
    if not isinstance(preds, np.ndarray):
        preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    t1 = np.sum(abs(preds-labels)) / np.size(labels)
    t2 = np.sum(abs(labels)) / np.size(labels)
    return t1 / t2

def get_acc(preds, labels):
    acc_list = []
    y_p = preds.reshape(-1)
    y_t = labels.reshape(-1)
    for a, b in zip(y_p, y_t):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)

def SMAPE(y_pred, y_true):
    s = 0
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a-b) / (abs(a) + abs(b))
    return s / np.size(y_true)

def MSPE(preds, labels):
    if not isinstance(preds, np.ndarray):
        preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    return np.mean(np.square((preds - labels) / labels))


def pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    x,y = x.reshape(-1),y.reshape(-1)
    return np.corrcoef(x,y)[0][1]


def node_pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    sigma_x = x.std(axis=0)
    sigma_y = y.std(axis=0)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    cor = ((x - mean_x) * (y - mean_y)).mean(0) / (sigma_x * sigma_y + 0.000000000001)
    return cor.mean()

def corr_np(preds, labels):
    sigma_p = (preds).std(axis=0)
    sigma_g = (labels).std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = labels.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((preds - mean_p) * (labels - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation


def stemgnn_mape(preds,labels, axis=None):
    '''
    Mean absolute percentage error.
    :param labels: np.ndarray or int, ground truth.
    :param preds: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mape = (np.abs(preds - labels) / (np.abs(labels)+1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def stnorm_mape(preds, labels):
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    greater_than_20 = labels > 20
    labels = labels[greater_than_20]
    preds = preds[greater_than_20]
    return np.mean(np.abs(preds - labels) / (labels))

class Evaluator(AbstractEvaluator):
    def __init__(self, config):
        self.config = config
        self.task = self.config.get("task", "")
        self.mask = self.config.get("mask", False)
        if self.task == "multi_step":
            self.out_catagory = "multi"
        elif self.task == "single_step":
            self.out_catagory = "single"
        else:
            raise ValueError('Can not find the type of evaluator!')


    def _evaluate(self, output:np.ndarray, groud_truth:np.ndarray, mask: int, out_catagory: str):
        """
        evluate the model performance
        : multi
        :param output: [n_samples, 12, n_nodes, n_features]
        :param groud_truth: [n_samples, 12, n_nodes, n_features]
        : single
        
        :return: dict [str -> float]
        """
        if out_catagory == 'multi':
            if bool(mask):
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims(groud_truth[...,0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
                batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                for step in range(steps):
                    # y_pred = np.reshape(output[:,step],(batch, -1))
                    # y_true = np.reshape(groud_truth[:,step],(batch,-1))
                    y_pred = output[:, step, ...]
                    y_true = groud_truth[:, step, ...]

                    scores['CORR'][f'horizon-{step}'] = corr_np(y_pred, y_true)
                    scores['MAE'][f'horizon-{step}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
                    scores['RMSE'][f'horizon-{step}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
                    scores['MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0)
                    scores['MAPE1'][f'horizon-{step}'] = mape_np(y_pred, y_true)
                    scores['node_wise_PCC'][f'horizon-{step}']= node_pcc_np(y_pred, y_true)
                    # scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                    # scores['StemGNN_MAPE'][f'horizon-{step}'] = stemgnn_mape(y_pred, y_true) * 100.0
                    # scores['CORR'][f'horizon-{step}'] = corr_np(y_pred, y_true)
                    # scores['CORR'][f'horizon-{step}'] = corr_np(y_pred.swapaxes(1,-1).reshape((-1,node)), y_true.swapaxes(1,-1).reshape((-1,node)))
                scores['MAE']['all'] = masked_mae_np(output,groud_truth, null_val=0.0)
                scores['RMSE']['all'] = masked_rmse_np(output,groud_truth, null_val=0.0)
                scores['MAPE']['all'] = masked_mape_np(output,groud_truth, null_val=0)
                scores["MAPE1"]['all'] = mape_np(output, groud_truth)
                # scores['PCC']['all'] = pcc_np(output,groud_truth)
                scores["node_wise_PCC"]['all'] = node_pcc_np(output, groud_truth)
                # scores['StemGNN_MAPE']['all'] = stemgnn_mape(output,groud_truth) * 100.0
                scores['CORR']['all'] = corr_np(output, groud_truth)
            else:
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims( groud_truth[...,0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
                batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                for step in range(steps):
                    # y_pred = output[:,step]
                    # y_true = groud_truth[:,step]
                    y_pred = output[:, step, ...]
                    y_true = groud_truth[:, step, ...]
                    scores['MAE'][f'horizon-{step}'] = mae_np(y_pred, y_true)
                    scores['RMSE'][f'horizon-{step}'] = rmse_np(y_pred, y_true)
                    # scores['MAPE'][f'horizon-{step}'] = mape_np(y_pred,y_true) * 100.0
                    scores['MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0)
                    # scores['STNorm_MAPE'][f'horizon-{step}'] = stnorm_mape(y_pred, y_true) * 100.0
                    # scores['StemGNN_MAPE'][f'horizon-{step}'] = stemgnn_mape(y_pred, y_true) * 100.0
                    # scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                    scores['node_wise_PCC'][f'horizon-{step}']= node_pcc_np(y_pred, y_true)
                    scores['CORR'][f'horizon-{step}'] = corr_np(y_pred, y_true)
                    scores["MSE"][f'horizon-{step}'] = mse_np(y_pred, y_true)
                scores['MAE']['all'] = mae_np(output,groud_truth)
                scores['RMSE']['all'] = rmse_np(output,groud_truth)
                scores['MAPE']['all'] = masked_mape_np(output,groud_truth, null_val=0)
                # scores['StemGNN_MAPE']['all'] = stemgnn_mape(output,groud_truth) * 100.0
                # scores['PCC']['all'] = pcc_np(output,groud_truth)
                scores['node_wise_PCC']['all'] = node_pcc_np(output, groud_truth)
                # scores['MSPE']['all'] = MSPE(output, groud_truth)
                scores['CORR']['all'] = corr_np(output, groud_truth)
                scores['MSE']['all'] = mse_np(output, groud_truth)
        else:
            output = output.squeeze()
            groud_truth = groud_truth.squeeze()
            assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
            scores = defaultdict(dict)

            scores['NRMSE']['all'] = nrmse(output, groud_truth)
            scores['ACC']['all'] = get_acc(output, groud_truth)
            scores['ND']['all'] = nd(output, groud_truth)
            scores['SMAPE']['all'] = SMAPE(output, groud_truth)
            scores['MSPE']['all'] = MSPE(output, groud_truth)

            scores['RMSE']['all'] = rmse_np(output, groud_truth)
            scores['MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.0)
            # scores['PCC']['all'] = node_pcc_np(output, groud_truth)
            scores['rse']['all'] = rse_np(output, groud_truth)
            scores['rae']['all'] = rae_np(output, groud_truth)
            # scores['MAPE']['all'] = stemgnn_mape(output, groud_truth) * 100.0
            scores['MAE']['all'] = mae_np(output, groud_truth)
            scores["node_pcc"]['all'] = node_pcc_np(output, groud_truth)
            scores['CORR']['all'] = corr_np(output, groud_truth)
        return scores


    def evaluate(self, output, groud_truth):
        if not isinstance(output, np.ndarray):
            output = output.clone().cpu().numpy()
        if not isinstance(groud_truth, np.ndarray):
            groud_truth = groud_truth.clone().cpu().numpy()
        return self._evaluate(output, groud_truth, self.mask, self.out_catagory)
