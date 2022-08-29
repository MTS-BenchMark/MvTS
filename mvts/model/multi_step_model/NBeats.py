# from _typeshed import Self
import torch
import torch.nn.functional as F
import torch.nn as nn


class NBeats(object):
    def __init__(self, config, data_feature):

        super(NBeats, self).__init__()

        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]
        self.num_batches = data_feature["num_batches"]
        

    def forward(self, x):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N, 2)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N, 2

        del need_concat

        return out



