import numpy as np
import torch
from torch import nn
class GCM(nn.Module):
    def __init__(self, input_dim):
        super(GCM, self).__init__()
        dmodel = input_dim * 12

        self.dmodel = dmodel
        self.relu = nn.ReLU()
        self.eps = 1e-6

        self.key = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)

        self.out = nn.Linear(dmodel, dmodel)

    def get_index(self, Nodes):
        index = np.pi / 2 * torch.arange(1, Nodes + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)

    def forward(self,x): # (B, T, N, D)
        B, T, N, D = x.shape

        # CLA
        x = x.transpose(1, 2).reshape(B, N, T * D)  # (B, N, T, D) -> (B, N, T*D)
        res = x

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.relu(q)*q
        k = self.relu(k)*k

        weight_index = self.get_index(N).to(x.device)
        # (B * h, N, 2 * d)

        q_ = torch.cat( [q * torch.sin(weight_index[:, :N, :] / N), q * torch.cos(weight_index[:, :N, :] / N)], dim=-1)
        k_ = torch.cat( [k * torch.sin(weight_index[:, :N, :] / N), k * torch.cos(weight_index[:, :N, :] / N)], dim=-1)

        # (B * h, N, 2 * d) (B * h, N, d) -> (B * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        # (B * h, N, 2 * d) (B * h, 2 * d) -> (B * h, N)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), self.eps)
        # (B * h, N, 2 * d) (B * h, d, 2 * d) (B * h, N) -> (B * h, N, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        # (B * h, N, d) -> (B * h, d, N) -> (B, D, N)-> (B, N, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, N).transpose(1, 2)
        # (B, N, D) -> (B, N, T, D) -> (B, N, D, T)

        attn_output = self.out(attn_output+res).reshape(B, N, T, D).transpose(1, 2)


        return attn_output

