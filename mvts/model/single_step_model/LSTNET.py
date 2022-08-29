import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNET(nn.Module):
    def __init__(self, config, data_feature):
        super(LSTNET, self).__init__()
        self.config = config
        self.use_cuda = self.config.get("cuda", True)
        self.P = self.config.get("window", 168)
        self.m = self.config.get("num_nodes", 0)
        self.m = self.m * self.config.get("input_dim", 1)
        self.hidR = self.config.get("hidRNN", 100)
        self.hidC = self.config.get("hidCNN", 100)
        self.hidS = self.config.get("hidSkip", 5)
        self.Ck = self.config.get("CNN_kernel", 6)
        self.skip = self.config.get("skip", 24)
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = self.config.get("highway_window", 24)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=self.config.get("dropout", 0.2))
        self.scaler = data_feature['scaler']
        
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        self.output_fun = self.config.get("output_fun", "sigmoid")
        if self.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if self.output_fun == 'tanh':
            self.output = F.tanh
 
    def forward(self, x):
        batch_size = x.size(0)
        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))
        # skip-rnn
        
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
            
        if self.output:
            res = self.output(res)
        return res
