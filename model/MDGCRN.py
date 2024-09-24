import pywt

from model.CLA import GCM
from model.MDGCRNCell import MDGCRNCell
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DGCC import DGCC


def disentangle(x, w, j=1):
    x = x.permute(0,3,2,1) # [S,D,N,T]
    x_np = x.cpu().numpy()
    coef = pywt.wavedec(x_np, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl_np = pywt.waverec(coefl, w)
    xh_np = pywt.waverec(coefh, w)
    # NumPy数组转回Tensor
    xl = torch.from_numpy(xl_np).to(x.device).permute(0, 3, 2, 1)
    xh = torch.from_numpy(xh_np).to(x.device).permute(0, 3, 2, 1)
    return xl, xh

class TemporalAttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttentionMechanism, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_length, num_nodes, hidden_dim]
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        value = self.value_layer(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_states.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, value)

        # 返回加权平均后的输出
        return torch.sum(weighted_values, dim=1)
class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_length, num_nodes, hidden_dim]
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        value = self.value_layer(hidden_states)

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.hidden_dim ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        weighted_values = torch.matmul(attention_weights, value)

        return weighted_values

class DGCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.embed_dim = embed_dim
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(MDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.DGCRM_cells.append(MDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        # self.gcm = GCM(node_num,dim_out)
        # self.temporal_attention = Attention(dim_out)  # Assuming seq_length is known
    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            if i==0:
                state = init_state[i]  # state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GRU的输入是前一层的隐藏状态
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :], node_embeddings[1]])#state=[batch,steps,nodes,input_dim]
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            # states_tensor = torch.stack(inner_states, dim=1)  # [batch_size, seq_length, num_nodes, hidden_dim]
            # states_tensor = self.temporal_attention(states_tensor)  # Apply the attention
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # current_inputs = self.temporal_attention(current_inputs)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)



class MDGCRN(nn.Module):
    def __init__(self, args):
        super(MDGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.embed_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.embed_dim))
        self.TIME = nn.Parameter(torch.empty(295, args.embed_dim))
        self.encoder1 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        self.encoder2 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        #predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * args.embed_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid())
        self.gcn = DGCC( self.input_dim , 2 * self.hidden_dim, args.cheb_k, args.embed_dim)
        self.gcm = GCM(self.output_dim)
        self.out = nn.Linear(128, self.output_dim)
        # self.temporal_attention = AttentionMechanism(self.output_dim)
    def forward(self, source, i=2):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        if self.use_D:
            t_i_d_data   = source[..., 1]#[B,T,N]
            T_i_D_emb1 = self.T_i_D_emb1[(t_i_d_data * 288).type(torch.LongTensor)]#[B,T,N,D]
            # node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data   = source[..., 2]
            D_i_W_emb1 = self.D_i_W_emb1[(d_i_w_data).type(torch.LongTensor)]#[B,T,N,D]
            # node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        source = source[..., 0].unsqueeze(-1)#[B,T,N,1]
        node_embedding1 = torch.cat((T_i_D_emb1,D_i_W_emb1), dim=-1)
        # node_embedding2 = torch.cat((T_i_D_emb2,D_i_W_emb2), dim=-1)
        # separation_coefficient = self.fc(node_embedding1/(node_embedding1+ node_embedding2))
        source1, source2 = disentangle(source,'coif1',)
        node_embeddings = [node_embedding1, self.node_embeddings1]  # [B,T,N,D]


        init_state1 = self.encoder1.init_hidden(source1.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        output1, _ = self.encoder1(source1, init_state1, node_embeddings)  # B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        output1 = self.dropout1(output1[:, -1:, :, :])
        output1 = self.end_conv1(output1)  # B, T*C, N, 1

        init_state2 = self.encoder2.init_hidden(source2.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        output2, _ = self.encoder2(source2, init_state2, node_embeddings)  # B, T, N, hidden
        # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
        output2 = self.dropout2(output2[:, -1:, :, :])
        output2 = self.end_conv2(output2)


        return self.alpha*output1 + (1-self.alpha)*output2
        # return output1


        # if i == 1:
        #     init_state1 = self.encoder1.init_hidden(source1.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        #     output1, _ = self.encoder1(source1, init_state1, node_embeddings)  # B, T, N, hidden
        #     # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #     output1 = self.dropout1(output1[:, -1:, :, :])
        #     output1 = self.end_conv1(output1)  # B, T*C, N, 1
        #
        #     # node_embeddings = [node_embedding2, self.node_embeddings2]
        #     init_state2 = self.encoder2.init_hidden(source2.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        #     output2, _ = self.encoder2(source2, init_state2, node_embeddings)  # B, T, N, hidden
        #     # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
        #     output2 = self.dropout2(output2[:, -1:, :, :])
        #     output2 = self.end_conv2(output2)
        #
        #     return output1 + output2
        #
        # else:
        #     init_state1 = self.encoder1.init_hidden(source1.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        #     output, _ = self.encoder1(source1, init_state1, node_embeddings)  # B, T, N, hidden
        #     # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #     output = self.dropout1(output[:, -1:, :, :])
        #     # CNN based predictor
        #     output1 = self.end_conv1(output)  # B, T*C, N, 1
        #     y_hat1 = self.end_conv2(output)
        #     y_error1 = source1 - y_hat1
        #     init_state2 = self.encoder2.init_hidden(y_error1.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        #     y_error_output1, _ = self.encoder2(y_error1, init_state2, node_embeddings)  # B, T, N, hidden
        #     y_error_output1 = self.dropout2(y_error_output1[:, -1:, :, :])
        #     y_error_output1 = self.end_conv3(y_error_output1)
        #
        #     output, _ = self.encoder1(source2, init_state1, node_embeddings)  # B, T, N, hidden
        #     # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #     output = self.dropout1(output[:, -1:, :, :])
        #     # CNN based predictor
        #     output2 = self.end_conv1(output)  # B, T*C, N, 1
        #     y_hat2 = self.end_conv2(output)
        #     y_error2 = source2 - y_hat2
        #     y_error_output2, _ = self.encoder2(y_error2, init_state2, node_embeddings)  # B, T, N, hidden
        #     y_error_output2 = self.dropout2(y_error_output2[:, -1:, :, :])
        #     y_error_output2 = self.end_conv3(y_error_output2)
        #
        #     return output1 + y_error_output1+output2 + y_error_output2


