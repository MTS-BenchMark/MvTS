import torch
import torch.nn as nn
import pandas as pd
import math
import numpy as np
import torch.nn.functional as F


class FcBlock(nn.Module):
    def __init__(self, config, input_size):
        super(FcBlock, self).__init__()
        self.config = config
        self.device = torch.device(self.config.get('device', "cpu"))
        self.input_size = input_size
        self.output_size = self.config.get('horizon')
        self.block_layers = self.config.get('block_layers')
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_size, self.config.get('hidden_units')))
        for i in range(1, self.block_layers):
            self.fc_layers.append(
                nn.Linear(self.config.get('hidden_units'), self.config.get('hidden_units')).to(self.device)
            )
        self.forecast = nn.Linear(self.config.get('hidden_units'), self.output_size).to(self.device)
        self.backcast = nn.Linear(self.config.get('hidden_units'), self.input_size).to(self.device)


    def forward(self, inputs):
        h = self.fc_layers[0](inputs) #[B, N, N*h+h+node_id_dim] -> [B, N, hidden_units]
        h = nn.functional.relu(h)
        for i in range(1, self.block_layers):
            h = self.fc_layers[i](h)
            h = nn.functional.relu(h)
        # h.shape:[batch_size, num_nodes, hidden_units]
        backcast = nn.functional.relu(inputs - self.backcast(h)) #[B, N, N*h+h+node_id_dim]
        return backcast, self.forecast(h)

class FcGagaLayer(nn.Module):
    def __init__(self, config, input_size):
        super(FcGagaLayer, self).__init__()
        self.config = config
        self.num_nodes = self.config.get('num_nodes') * (self.config.get('output_dim'))
        self.device = torch.device(self.config.get('device', "cpu"))
        self.window_size = self.config.get('window_size')
        self.input_size = input_size
        self.output_size = self.config.get('horizon')
        self.node_id_dim = self.config.get('node_id_dim')
        self.epsilon = self.config.get('epsilon')
        self.blocks_num = self.config.get('blocks')

        # self.blocks = []
        # for i in range(self.blocks_num):
        #     self.blocks.append(FcBlock(config, input_size=self.input_size))
        self.blocks = nn.ModuleList([FcBlock(config, input_size=self.input_size)
                                     for i in range(self.blocks_num)])


        self.node_id_em = nn.Embedding(num_embeddings=self.num_nodes,
                                       embedding_dim=self.node_id_dim).to(self.device)

        self.time_gate1 = nn.Linear(self.node_id_dim+self.window_size, self.config.get('hidden_units')).to(self.device)

        self.time_gate2 = nn.Linear(self.config.get('hidden_units'), self.config.get('horizon')).to(self.device)
        self.time_gate3 = nn.Linear(self.config.get('hidden_units'), self.config.get('window_size')).to(self.device)

    def forward(self, history_in, node_id_in, time_of_day_in):
        node_id = self.node_id_em(node_id_in) #[batch_size, num_nodes]->[batch_size, num_nodes, node_id_dim]
        node_embeddings = np.squeeze(node_id[0, :, :]) #[num_nodes, nodes_id_dim]
        # node_id = tf.squeeze(node_id, axis=-2)
        # node_id = np.squeeze(node_id, axis=-2) #[batch_size, num_nodes, nodes_id_dim]
        time_gate = self.time_gate1(torch.cat((node_id, time_of_day_in), dim=-1)) #[batch_size, num_nodes, node_id_dim+window_size]->[batch_size, num_nodes, hidden_units]
        time_gate = nn.functional.relu(time_gate) #[batch_size, num_nodes, hidden_units]
        time_gate_forward = self.time_gate2(time_gate) #[batch_size, num_nodes, horizon]
        time_gate_backward = self.time_gate3(time_gate)

        history_in = history_in / (1.0 + time_gate_backward) #[batch_size, num_nodes, horizon]

        node_embeddings_dp = torch.mm(node_embeddings, node_embeddings.transpose(1, 0)) #[num_nodes, num_nodes]
        node_embeddings_dp = torch.exp(self.epsilon * node_embeddings_dp)
        node_embeddings_dp = node_embeddings_dp.unsqueeze(0)
        node_embeddings_dp = node_embeddings_dp.unsqueeze(-1) #[1, num_nodes, num_nodes, 1]

        level = np.max(history_in.cpu().detach().numpy(), axis=-1, keepdims=True) #[batch_size, num_nodes, 1]
        level = torch.from_numpy(level).to(self.device)
        history = torch.divide(history_in, level)
        where_is_inf = history.isinf()
        history[where_is_inf] = 0 #[batch_size, num_nodes, horizon]
        # history = torch.where(history==np.inf, 0, history) #[batch_size, num_nodes, horizon]
        # Add history of all other nodes
        all_node_history = history_in.unsqueeze(1).repeat(1, self.num_nodes, 1, 1) #[batch_size, num_nodes, num_nodes, horizon]

        all_node_history = all_node_history * node_embeddings_dp #[B, N, N, h]
        all_node_history = all_node_history.reshape(-1, self.num_nodes, self.num_nodes * history_in.shape[2]) #[B, N, N*h]
        all_node_history = torch.divide(all_node_history - level, level)
        where_is_inf = all_node_history.isinf()
        all_node_history[where_is_inf] = 0
        all_node_history = all_node_history.to(torch.float32)
        all_node_history = nn.functional.relu(all_node_history)
        # all_node_history = torch.where(all_node_history > 0.0, all_node_history, 0.0)
        history = torch.cat((history, all_node_history), dim=-1) #[B, N, N*h+h]
        # Add node ID
        history = torch.cat((history, node_id), dim=-1) #[B, N, N*h+h+node_id_dim]

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.blocks_num):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:, :, :self.output_size]
        forecast = forecast_out * level

        forecast = (1.0 + time_gate_forward) * forecast

        return backcast, forecast

class FCGAGA(nn.Module):
    def __init__(self, config, data_feature):
        super(FCGAGA, self).__init__()

        self.data_feature = data_feature
        self.config = config
        self.device = torch.device(self.config.get('device', "cpu"))
        self.scaler = self.data_feature['scaler']
        self.node_id = self.data_feature['node_id']

        self.num_nodes = self.config.get('num_nodes') * (self.config.get('output_dim'))
        self.window_size = self.config.get('window_size')
        self.node_id_dim = self.config.get('node_id_dim')
        self.num_stacks = self.config.get('num_stacks')

        self.horizon = self.config.get('horizon')
        self.input_size = self.window_size + self.node_id_dim + self.num_nodes * self.window_size

        # self.fcgaga_layers = []
        # for i in range(self.num_stacks):
        #     self.fcgaga_layers.append(FcGagaLayer(self.config, self.input_size))
        self.fcgaga_layers = nn.ModuleList([FcGagaLayer(self.config, self.input_size)
                                            for i in range(self.num_stacks)])
        self.to(self.device)

    def forward(self, history_in, time_of_day_in, node_id_in):
        node_id_in = torch.unsqueeze(node_id_in, dim=0).repeat(history_in.shape[0], 1)
        history_in = history_in.permute(0, 2, 1).to(torch.float32)
        time_of_day_in = time_of_day_in.permute(0, 2, 1).to(torch.float32)
        # history_in[batch_size, window_size, num_nodes], expect[batch_size, num_nodes, window_size]
        # time_of_day_in[batch_size, window_size, num_nodes], expect[batch_size, num_nodes, window_size]
        #node_id_in[batch_size, num_nodes]
        backcast, forecast = self.fcgaga_layers[0](history_in=history_in, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
        for nbg in self.fcgaga_layers[1:]:
            backcast, forecast_graph = nbg(history_in=forecast, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
            forecast = forecast + forecast_graph
        forecast = forecast / self.num_stacks #[B, N, horizon]
        where_is_nan = forecast.isnan()
        forecast[where_is_nan] = 0
        # forecast = torch.where(math.isnan(forecast), torch.zeros_like(forecast), forecast)
        forecast = forecast.permute(0, 2, 1)
        return forecast