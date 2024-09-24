import torch
import torch.nn as nn

from model.DGCC import DGCC


class MDGCRNCell(nn.Module):  #这个模块只进行GRU内部的更新，所以需要修改的是AGCN里面的东西
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(MDGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        # self.gate = DGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        # self.update = DGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.gate = DGCC(dim_in + self.hidden_dim + 2 * embed_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = DGCC(dim_in + self.hidden_dim + 2 * embed_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, 1(速度)
        #state: B, num_nodes, hidden_dim=64
        #node_embeddings[0]:B,num_nodes,2*embed_dim(embed_dim=10)
        #node_embeddings[1]:num_nodes,embed_dim=10
        state = state.to(x.device)
        input_and_state_and_time= torch.cat((x, state,node_embeddings[0]), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state_and_time, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)#Z,R=[B,N,hidden_dim]
        candidate = torch.cat((x, z*state,node_embeddings[0]), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
