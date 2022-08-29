import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TPA_LSTM(nn.Module):
    def __init__(self, config, data_feature):
        super(TPA_LSTM, self).__init__()

        self.config = config
        self.data_feature = data_feature

        self.use_cuda = self.config.get("cuda", True)
        self.window_length = self.config.get("window", 168)  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        
        self.original_columns = self.config.get("num_nodes", 0)  # the number of columns or features
        self.original_columns = self.original_columns * self.config.get("input_dim",  1)

        self.hidR = self.config.get("hidRNN", 100)

        self.hidden_state_features = self.config.get("hidden_state_features", 140)
        self.hidC = self.config.get("hidCNN", 100)

        self.hidS = self.config.get("hidSkip", 5)
        self.Ck = self.config.get("CNN_kernel", 168)  # the kernel size of the CNN layers
        self.skip = self.config.get("skip", 24)

        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = self.config.get("highway_window", 24)
        
        self.num_layers_lstm = self.config.get("num_layers_lstm", 1)

        self.scaler = data_feature['scaler']
        
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False)
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, 1))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(torch.randn(self.hidC, self.hidden_state_features), requires_grad=True)
        self.context_vector_linear = nn.Linear(self.hidC, self.hidden_state_features, bias=False)
        self.final_state_linear = nn.Linear(self.hidden_state_features,self.hidden_state_features,bias=False)
        self.final_hn_linear = nn.Linear(self.hidden_state_features, self.original_columns,bias=False)
        # self.attention_matrix = nn.Parameter(
        #     torch.randn(args.batch_size, self.hidC, self.hidden_state_features, requires_grad=True, device='cuda'))
        # self.context_vector_matrix = nn.Parameter(
        #     torch.randn(args.batch_size, self.hidden_state_features, self.hidC, requires_grad=True, device='cuda'))
        # self.final_state_matrix = nn.Parameter(
        #     torch.randn(args.batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True, device='cuda'))
        # self.final_matrix = nn.Parameter(
        #     torch.randn(args.batch_size, self.original_columns, self.hidden_state_features, requires_grad=True, device='cuda'))
        # torch.nn.init.xavier_uniform(self.attention_matrix)
        # torch.nn.init.xavier_uniform(self.context_vector_matrix)
        # torch.nn.init.xavier_uniform(self.final_state_matrix)
        # torch.nn.init.xavier_uniform(self.final_matrix)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns))  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=self.config.get("dropout", 0.2))
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns)
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        output_fun = self.config.get("output_fun", "sigmoid")
        if (output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (output_fun == 'tanh'):
            self.output = F.tanh
        if (output_fun == 'relu'):
            self.output = F.leaky_relu

    def forward(self, input):
        #print('???????????????????????')
        #print(self.original_columns) #(32, 24, 28) (32, 168, 28)
        #print(input.shape)
        batch_size = input.size(0)
        if (self.use_cuda):
            x = input.cuda()

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states 

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        input_to_lstm = x.permute(1, 0, 2).contiguous()  # input to lstm is of shape (seq_len, batch, input_size) (x shape (batch_size, seq_length, features))
        #print(input_to_lstm)
        # print(input_to_lstm.shape)
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        # print('h_all!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(h_all)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))
        
        # print('hn')
        # print(hn)

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features)
        #print("((((((((((((((((((((((")
        #print(input_to_convolution_layer.shape) # (32, 1, 24, 140) (32, 1, 168, 140) 
        convolution_output = F.leaky_relu(self.compute_convolution(input_to_convolution_layer))
        #print("((((((((((((((((((((((")
        #print(self.compute_convolution(input_to_convolution_layer).shape) # (32, 10, 1, 140) (32, 10, 145, 140)
        convolution_output = self.dropout(convolution_output)
        #print("((((((((((((((((((((((")
        #print(convolution_output.shape) # (32, 10, 1, 140) (32, 10, 145, 140)
        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        # final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        # # print(final_hn.shape)
        # # print("hn")
        # # print(hn.shape)
        # final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        # # print(convolution_output.shape)
        # diff = 0
        # if (hn.size(0) < self.attention_matrix.size(0)):
        #     final_hn[:hn.size(0), :, :] = hn
        #     final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
        #     diff = self.attention_matrix.size(0) - hn.size(0)
        # else:
        #     final_hn = hn
        #     final_convolution_output = convolution_output

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        #print("after")
        #print(convolution_output.shape)
        convolution_output, hn = convolution_output.squeeze(2).transpose(1,2), hn.transpose(1,2)
        #print("____________________")
        #print(convolution_output.shape, hn.shape, self.attention_matrix.shape)
        mat = torch.matmul(convolution_output,self.attention_matrix)
        scores = mat.bmm(hn)
        alpha = torch.sigmoid(scores)
        context_vector = torch.sum(alpha * convolution_output, dim=1)
        hn = hn.squeeze()



        final_hn = self.context_vector_linear(context_vector) + self.final_state_linear(hn)
        output = self.final_hn_linear(final_hn)
        # output = self.output(output)
        return output
        # print(context_vector.shape, )



        # convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        # final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        # # print('  convolution_output_for_scoring')
        # # print(convolution_output_for_scoring)
        # convolution_output_for_scoring = convolution_output_for_scoring.cuda()
        # final_hn_realigned = final_hn_realigned.cuda()
        # mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        # scoring_function = torch.bmm(mat1, final_hn_realigned)
        # alpha = torch.nn.functional.sigmoid(scoring_function)
        # # print(alpha.shape, convolution_output_for_scoring.shape)
        # context_vector = alpha * convolution_output_for_scoring
        # context_vector = torch.sum(context_vector, dim=1)
        # # print('contex_vector')
        #
        # # print(context_vector)
        #
        # """
        #    Step 4. Compute the output based upon final_hn_realigned, context_vector
        # """
        # context_vector = context_vector.view(-1, self.hidC, 1)
        # h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)
        # # print('step4')
        # # print(self.final_matrix)
        # # print(h_intermediate)
        # result = torch.bmm(self.final_matrix, h_intermediate)
        # result = result.permute(0, 2, 1).contiguous()
        # result = result.squeeze()
        # # print('result')
        # # print(result)
        #
        # #result = self.linear1(result)
        #
        #
        # """
        #    Remove from result the extra result points which were added as a result of padding
        # """
        # final_result = result[:result.size(0) - diff]
        #
        # """
        # Adding highway network to it
        # """
        # # print('final_result  shape!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # # print(final_result.shape)
        #
        # res = self.output(final_result)
        # #print(res.shape)
        #
        # # if (self.hw > 0):
        # #     z = x[:, -self.hw:, :];
        # #     z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
        # #     z = self.highway(z);
        # #     z = z.view(-1, self.original_columns);
        # #     res = final_result + z;
        # #
        # # if self.output:
        # #     res = self.output(res)
        # # print(res.shape)
        # return res
        #torch.sigmoid(res)

