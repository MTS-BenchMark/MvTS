import importlib
import numpy as np
import copy
import pickle
from libcity.model import loss
from functools import partial

def get_train_loss(train_loss):
    """
    get the loss func
    """
    if train_loss.lower() == 'none':
        print('Warning. Received none train loss func and will use the loss func defined in the model.')
        return None
    
    def func(preds, labels):

        if train_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif train_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif train_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif train_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif train_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif train_loss.lower() == 'huber':
            lf = loss.huber_loss
        elif train_loss.lower() == 'quantile':
            lf = loss.quantile_loss
        elif train_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif train_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif train_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif train_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif train_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif train_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch

        return lf(preds, labels)
    return func

