import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import datetime
import h5py
import pandas as pd
import numpy as np
from mvts.data.dataset import AbstractDataset
from torch.utils.data import Dataset, DataLoader

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss

def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss

def cosine(source, target):
    source, target = source.mean(), target.mean()
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()

def kl_div(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    criterion = nn.KLDivLoss(reduction='batchmean')
    loss = criterion(source.log(), target)
    return loss

def js(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    M = .5 * (source + target)
    loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
    return .5 * (loss_1 + loss_2)

class Mine_estimator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine_estimator, self).__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X, Y):
        Y_shffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shffle)
        ret = torch.mean(loss_joint) - \
            torch.log(torch.mean(torch.exp(loss_marginal)))
        loss = -ret
        return loss


class Mine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2

def adv(source, target, input_dim=256, hidden_dim=512):
    domain_loss = nn.BCELoss()
    # !!! Pay attention to .cuda !!!
    adv_net = Discriminator(input_dim, hidden_dim).cuda()
    domain_src = torch.ones(len(source)).cuda()
    domain_tar = torch.zeros(len(target)).cuda()
    domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
    reverse_src = ReverseLayerF.apply(source, 1)
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    pred_tar = adv_net(reverse_tar)
    loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s + loss_t
    return loss

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def pairwise_dist(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = X.unsqueeze(1).expand(n, m, d)
    b = Y.unsqueeze(0).expand(n, m, d)
    return torch.pow(a - b, 2).sum(2)

class TransferLoss(object):
    def __init__(self, loss_type='cosine', input_dim=512):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == 'mmd_lin' or self.loss_type =='mmd':
            mmdloss = MMD_loss(kernel_type='linear')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'coral':
            loss = CORAL(X, Y)
        elif self.loss_type == 'cosine' or self.loss_type == 'cos':
            loss = 1 - cosine(X, Y)
        elif self.loss_type == 'kl':
            loss = kl_div(X, Y)
        elif self.loss_type == 'js':
            loss = js(X, Y)
        elif self.loss_type == 'mine':
            mine_model = Mine_estimator(
                input_dim=self.input_dim, hidden_dim=60).cuda()
            loss = mine_model(X, Y)
        elif self.loss_type == 'adv':
            loss = adv(X, Y, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == 'mmd_rbf':
            mmdloss = MMD_loss(kernel_type='rbf')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'pairwise':
            pair_mat = pairwise_dist(X, Y)
            import torch
            loss = torch.norm(pair_mat)

        return loss



def TDC(num_domain, data_file, dis_type='coral'):
    start_time = datetime.datetime.strptime(
        '2013-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(
        '2016-06-30 23:00:00', '%Y-%m-%d %H:%M:%S')
    num_day = (end_time - start_time).days
    split_N = 10
    # data = pd.read_pickle(data_file)[station]
    f = h5py.File(data_file, "r")
    feat = np.array(f["raw_data"])
    feat = feat[0:num_day]
    feat = torch.tensor(feat, dtype=torch.float32)
    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]), hours=0)
            else:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]) + 1,
                                                                 hours=0)
            sel_end_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i]), hours=23)
            sel_start_time = datetime.datetime.strftime(sel_start_time, '%Y-%m-%d %H:%M')
            sel_end_time = datetime.datetime.strftime(sel_end_time, '%Y-%m-%d %H:%M')
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")

def get_split_time(num_domain=2, mode='pre_process', data_file = None, dis_type = 'coral'):
    spilt_time = {
        '2': [('2013-3-6 0:0', '2015-5-31 23:0'), ('2015-6-2 0:0', '2016-6-30 23:0')]
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    if mode == 'tdc':
        return TDC(num_domain, data_file, dis_type=dis_type)
    else:
        print("error in mode")

class data_loader(Dataset):
    def __init__(self, df_feature, df_label, df_label_reg, t=None):

        assert len(df_feature) == len(df_label)
        assert len(df_feature) == len(df_label_reg)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature=df_feature
        self.df_label=df_label
        self.df_label_reg = df_label_reg

        self.T=t
        self.df_feature=torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label=torch.tensor(
            self.df_label, dtype=torch.float32)
        self.df_label_reg=torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        sample, target, label_reg =self.df_feature[index], self.df_label[index], self.df_label_reg[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target, label_reg

    def __len__(self):
        return len(self.df_feature)

def create_dataset(feat, label, label_reg, times, start_date, end_date, mean=None, std=None):
    # referece_start_time=datetime.datetime(2013, 3, 1, 0, 0)
    # referece_end_time=datetime.datetime(2017, 2, 28, 0, 0)
    referece_start_time = times[0]
    referece_end_time = times[-1]

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]

    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return data_loader(feat, label, label_reg)

def get_weather_data(data_file, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    # df=pd.read_pickle(data_file)
    f = h5py.File(data_file, "r")
    feat = np.array(f["raw_data"])
    label = np.array(f["label"])
    label_reg = np.array(f["label_reg"])
    time = np.array(f["time"])
    t = []
    for i in range(time.shape[0]):
        t.append(time[i].decode())
    time = np.stack(t, axis=0)
    time = pd.to_datetime(time)
    dataset = create_dataset(feat, label, label_reg, time, start_time,
                             end_time, mean=mean, std=std)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def get_dataset_statistic(feat, label, label_reg, times, start_date, end_date):
    # referece_start_time = datetime.datetime(2013, 3, 1, 0, 0)
    # referece_end_time = datetime.datetime(2017, 2, 28, 0, 0)

    referece_start_time = times[0]
    referece_end_time = times[-1]

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start = (pd.to_datetime(start_date) - referece_start_time).days
    index_end = (pd.to_datetime(end_date) - referece_start_time).days
    feat = feat[index_start: index_end + 1]
    label = label[index_start: index_end + 1]
    feat = feat.reshape(-1, feat.shape[2])
    mu_train = np.mean(feat, axis=0)
    sigma_train = np.std(feat, axis=0)

    return mu_train, sigma_train

def get_weather_data_statistic(data_file, start_time, end_time):
    f = h5py.File(data_file, "r")
    feat = np.array(f["raw_data"])
    label = np.array(f["label"])
    label_reg = np.array(f["label_reg"])
    time = np.array(f["time"])
    t = []
    for i in range(time.shape[0]):
        t.append(time[i].decode())
    time = np.stack(t, axis=0)
    time = pd.to_datetime(time)
    # df=pd.read_pickle(data_file)
    mean_train, std_train =get_dataset_statistic(
        feat, label, label_reg, time, start_time, end_time)
    return mean_train, std_train

class AdaRNNDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.seed = self.config.get("seed")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        file_name = self.config.get("filename", "")
        train_start_time = self.config.get("train_start_time")
        train_end_time = self.config.get("train_end_time")
        valid_start_time = self.config.get("valid_start_time")
        valid_end_time = self.config.get("valid_end_time")
        test_start_time = self.config.get("test_start_time")
        test_end_time = self.config.get("test_end_time")

        station = self.config.get("station", "Dingling")
        number_domain = self.config.get("number_domain", 10)
        data_mode = self.config.get("data_mode")
        batch_size = self.config.get("batch_size", 16)
        dis_type = "coral"

        mean_train, std_train = get_weather_data_statistic(file_name, start_time=train_start_time, end_time=train_end_time)
        split_time_list = get_split_time(number_domain, mode=data_mode, data_file=file_name, dis_type=dis_type)
        train_list = []
        for i in range(len(split_time_list)):
            time_temp = split_time_list[i]
            train_loader = get_weather_data(file_name, start_time=time_temp[0],
                                                         end_time=time_temp[1], batch_size=batch_size, mean=mean_train,
                                                         std=std_train)
            train_list.append(train_loader)

        self.valid_loader = get_weather_data(file_name, start_time=valid_start_time,
                                                         end_time=valid_end_time, batch_size=batch_size,
                                                         mean=mean_train, std=std_train)
        self.test_loader = get_weather_data(file_name, start_time=test_start_time,
                                                    end_time=test_end_time, batch_size=batch_size, mean=mean_train,
                                                    std=std_train, shuffle=False)
        self.train_loader = train_list


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集

        return self.train_loader,  self.valid_loader,  self.test_loader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {}

        return feature
