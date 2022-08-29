import math
import random
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10, device=None):
    sample = sample_gumbel(logits.size(), eps=eps, device = device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10, device=None):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps, device=device)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y



class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int):
        super(DCGRUCell, self).__init__()
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = GraphConv_(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop)
        self.candidate_g_conv = GraphConv_(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop)

    def forward(self, inputs: Tensor, supports: List[Tensor], states) -> Tuple[Tensor, Tensor]:
        """
        :param inputs: Tensor[Batch, Node, Feature]
        :param supports:
        :param states:Tensor[Batch, Node, Hidden_size]
        :return:
        """
        r_u = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c
        return outputs, new_state


class DCRNNEncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int, n_layers: int):
        super(DCRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.append(DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))

    def forward(self, inputs: Tensor, supports: List[Tensor]) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, input_size]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [n_layers, B, N, hidden_size]
        """

        b, t, n, _ = inputs.shape
        dv, dt = inputs.device, inputs.dtype

        states = list(torch.zeros(len(self), b, n, self.hidden_size, device=dv, dtype=dt))
        inputs = list(inputs.transpose(0, 1))

        for i_layer, cell in enumerate(self):
            for i_t in range(t):
                inputs[i_t], states[i_layer] = cell(inputs[i_t], supports, states[i_layer])

        return torch.stack(states)


class DCRNNDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, num_node: int,
                 n_supports: int, k_hop: int, n_layers: int, n_preds: int):
        super(DCRNNDecoder, self).__init__()
        self.output_size = output_size
        self.n_preds = n_preds
        self.append(DCGRUCell(output_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, supports: List[Tensor], states: Tensor,
                targets: Tensor = None, teacher_force: bool = 0.5) -> Tensor:
        """
        :param supports: list of sparse tensors, each of shape [N, N]
        :param states: tensor, [n_layers, B, N, hidden_size]
        :param targets: None or tensor, [B, T, N, output_size]
        :param teacher_force: random to use targets as decoder inputs
        :return: tensor, [B, T, N, output_size]
        """
        n_layers, b, n, _ = states.shape

        inputs = torch.zeros(b, n, self.output_size, device=states.device, dtype=states.dtype)

        states = list(states)
        assert len(states) == n_layers

        new_outputs = list()
        for i_t in range(self.n_preds):
            for i_layer in range(n_layers):
                inputs, states[i_layer] = self[i_layer](inputs, supports, states[i_layer])
            inputs = self.out(inputs)
            new_outputs.append(inputs)
            if targets is not None and random.random() < teacher_force:
                inputs = targets[:, i_t]

        return torch.stack(new_outputs, 1)


class GTS(nn.Module):

    def __init__(self, config, data_feature):
        super(GTS, self).__init__()

        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]
        self.num_batches = data_feature["num_batches"]

        n_pred = self.config.get("horizon", 12)
        hidden_size = self.config.get("model_hidden_size", 25)
        num_nodes = self.config.get("num_nodes", None)
        n_dim = self.config.get("n_dim", 50)
        n_supports = self.config.get("n_supports", 1)
        k_hop = self.config.get("k_hop", 3)
        n_rnn_layers = self.config.get("n_rnn_layers", 1)
        n_gconv_layers = self.config.get("n_gconv_layers", 3)
        input_dim = self.config.get("input_dim", 1)
        output_dim = self.config.get("output_dim", 1)
        cl_decay_steps = self.config.get("cl_decay_steps", 300)
        support = torch.from_numpy(self.data_feature["matrix"]).float()
        device = torch.device(self.config.get('device', "cpu"))
        self.cl_decay_steps = cl_decay_steps

        self.node_num = support.shape[1]
        # T, dim, N = support.shape
        # self.node_num = N
        # self.node_fea = support.permute(2, 0, 1).reshape(N, 1, -1)  # (T, DIM ,N)
        # self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        # self.conv2 = torch.nn.Conv1d(8, 4, 10, stride=5)  # .to(device)
        # self.hidden_drop = torch.nn.Dropout(0.2)
        # self.fc = torch.nn.Linear(dim_fc, embedding_dim)
        # self.bn1 = torch.nn.BatchNorm1d(8)
        # self.bn2 = torch.nn.BatchNorm1d(4)
        # self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.fc_out = nn.Linear(embedding_dim * 2, embedding_dim)
        # self.fc_cat = nn.Linear(embedding_dim, 2)

        self.device = device

        self.encoder = DCRNNEncoder(input_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers)
        self.decoder = DCRNNDecoder(output_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers, n_pred)


        # self.graph = nn.Parameter(support, requires_grad=True)
        self.graph = support.to(device)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.node_num, self.node_num])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)


    def forward(self, inputs: Tensor, targets: Tensor = None, batch_seen: int = None,temp=1) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, output_dim],[]
        """

        # x = self.conv1(self.node_fea)
        # x = F.relu(x)
        # x = self.bn1(x)
        # # print(x.shape)
        # # x = self.hidden_drop(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.bn2(x)
        # # print(x.shape)
        # x = x.reshape(self.node_num, -1)
        # x = self.fc(x)
        # x = F.relu(x)
        # x = self.bn3(x)
        # receivers = torch.matmul(self.rel_rec, x)
        # senders = torch.matmul(self.rel_send, x)
        # x = torch.cat([senders, receivers], dim=1)
        # x = torch.relu(self.fc_out(x))
        # x = self.fc_cat(x)
        graph = self.graph


        # di = self.graph.reshape(-1)
        # x = torch.stack([di,1-di], dim=1)
        #
        #
        #
        # graph = gumbel_softmax(x, temperature=temp, hard=False, device=self.device)
        # graph = graph[:, 0].clone().reshape(self.node_num, -1)
        # print(torch.sum(graph,dim=1))
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        # mask = torch.eye(self.node_num, self.node_num).bool().to(self.device)
        # graph.masked_fill_(mask, 0)

        states = self.encoder(inputs, graph)
        outputs = self.decoder(graph, states, targets, self._compute_sampling_threshold(batch_seen))
        return outputs, graph

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
        # return 0


class GraphConv_(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv_, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape
        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = x0.unsqueeze(dim=0)  # (1, num_nodes, input_dim * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = supports.mm(x0)
            x = self._concat(x, x1)
            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * supports.mm(x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)


class GraphConv_MX(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv_MX, self).__init__()
        self._num_nodes = num_nodes
        self.out = nn.Linear(input_dim * n_supports, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape
        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = list()

        for i in supports:
            support = self.Matrix_Normalization(i)
            x1 = support.mm(x0)
            x.append(x1)

        x = torch.stack(x, 0)
        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)

    def Matrix_Normalization(self, support):
        dv, dt = support.device, support.dtype
        n, m = support.shape
        x = support + torch.eye(n, device=dv, dtype=dt)
        # sum_x = torch.sum(x,1)
        # d = torch.eye(n,device=dv, dtype=dt) * sum_x ** -0.5
        # x = d.mm(x).mm(d)
        return x
