from collections import OrderedDict
from os.path import join
import math
import pdb

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import LayerNorm, GELU
from torch_geometric.nn import GENConv, DeepGCNLayer, GINConv
from torch.nn import Sequential as Seq

class PatchGCN_module(torch.nn.Module):
    def __init__(self, hidden_dim, i, dropout_rate):
        super(PatchGCN_module, self).__init__()
        self.conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
        self.norm = LayerNorm(hidden_dim, elementwise_affine=True)
        self.act = GELU()
        self.layer = DeepGCNLayer(self.conv, self.norm, self.act, block='res', dropout=0.1, ckpt_grad=i % 3)
        self.dropout_rate = dropout_rate
    def forward(self, x, edge_index, ):
        x_after = self.layer(x, edge_index)
        return x_after

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# GNN分batch
def pad_batch(data, batch, max_input_len=np.inf, get_mask=True, seed=None):
    """
        input:
        h_node: node features (N,dim)
        batch: batch information from pyg (N,1)
        max_input_len: max node num per batch (1)

        output:
        padded_h_node: batched features (b,n,dim)
        src_padding_mask: (b,n) 标记哪几个节点是有效节点

        num_nodes: 节点数量
        masks: bool mask list [(n,1), (n,1)]
        max_num_nodes
    """
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)  # n*1 bool掩码
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)
    # print('num_nodes', num_nodes)

    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = data.new(num_batch, max_num_nodes, data.size(-1)).fill_(0)  # b * n * k
    src_padding_mask = data.new(num_batch, max_num_nodes).fill_(0).bool()  # b * n

    for i, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed + i)
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            # 随机采样node
            random_indices = torch.randperm(num_node)
            selected_indices = random_indices[:max_num_nodes]
            selected_indices.sort()
            padded_h_node[i, :max_num_nodes] = data[mask][selected_indices]
        else:
            padded_h_node[i, :num_node] = data[mask][:num_node]
        src_padding_mask[i, :num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


def pad_batch_ordered(data, batch, max_input_len=np.inf, get_mask=True):
    """
        input:
        h_node: node features (N,dim)
        batch: batch information from pyg (N,1)
        max_input_len: max node num per batch (1)

        output:
        padded_h_node: batched features (b,n,dim)
        src_padding_mask: (b,n) 标记哪几个节点是有效节点

        num_nodes: 节点数量
        masks: bool mask list [(n,1), (n,1)]
        max_num_nodes
    """
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)  # n*1 bool掩码
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = data.new(num_batch, max_num_nodes, data.size(-1)).fill_(0)  # b * n * k
    src_padding_mask = data.new(num_batch, max_num_nodes).fill_(0).bool()  # b * n

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            padded_h_node[i, :max_num_nodes] = data[mask][:max_num_nodes]
        else:
            padded_h_node[i, :num_node] = data[mask][:num_node]
        src_padding_mask[i, :num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


def recover_batch(data_batched, src_padding_mask=None):
    n = data_batched.shape[0]

    ans_data = []
    batch = []

    for i in range(n):
        tmp_data = data_batched[i]
        if src_padding_mask is not None:
            tmp_shape = src_padding_mask[i]
            tmp_data = tmp_data[tmp_shape]
        tmp_batch = tmp_data.new(tmp_data.shape[0]).fill_(i).to(dtype=torch.int64)
        ans_data.append(tmp_data)
        batch.append(tmp_batch)

    ans_data = torch.cat(ans_data, dim=0)
    batch = torch.cat(batch, dim=0)

    return ans_data, batch


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x, attn_mask: torch.Tensor = None, valid_input_mask: torch.Tensor = None, mask_value=-1e6):
        """mask should be a 3D tensor of shape (B, T, T)"""
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_mask is not None:
            att = att.masked_fill(attn_mask.unsqueeze(1) == 0, mask_value)
        if valid_input_mask is not None:
            att = att.masked_fill(valid_input_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
