import torch
from torch.nn import init, GELU
from models.utils.model_utils import *

mask_value = float('-inf')

class ProtoGNN(torch.nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512,
                 dim_proto=512, nr_types=6, num_proto=8,
                 iters=2, n_head=8,
                 pre_proto=False, pre_proto_pth=None
                 ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_proto = num_proto
        self.nr_types = nr_types
        self.dim_proto = dim_proto
        self.pre_proto = pre_proto

        self.proto_attention = ProtoAttention(num_proto=num_proto, num_classes=nr_types, iters=iters,
                                              input_dim=self.feature_dim, hidden_dim=hidden_dim, dim_proto=dim_proto,
                                              n_head=n_head)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proto_mlp = nn.Linear(dim_proto, hidden_dim)

        try:
            print('pre_proto_pth', pre_proto_pth)
            self.pre_prototypes = torch.load(pre_proto_pth)
        except Exception as e:
            print(e)
            self.pre_prototypes = None

    def get_prototypes(self, x_batch, batch_mask, node_type, batch, batch_num):  # (N,C), (N)
        # x_batch b*n*1024
        proto_mu = torch.zeros((batch_num, self.nr_types, self.feature_dim)).to(x_batch.device)  # (3,1024)->(b,3,1024)

        if self.pre_proto:
            for i in range(self.nr_types):
                proto_mu[:, i] = self.pre_prototypes[i]
        else:
            for bth in range(batch_num):
                for i in range(self.nr_types):  # Class
                    type_mask = (node_type[bth] == i)
                    mask = batch_mask[bth] * type_mask
                    # drop aug nodes
                    non_zero_rows = (x_batch[bth] != 0).any(dim=1)
                    mask *= non_zero_rows
                    mask = mask.unsqueeze(-1)

                    if torch.sum(mask) == 0:
                        if self.pre_prototypes is not None:
                            proto_mu[bth, i] = self.pre_prototypes[i]
                        else:
                            proto_mu[bth, i] = torch.sum(x_batch[bth], dim=0) / torch.sum(batch_mask[bth])
                    else:
                        proto_mu[bth, i] = torch.sum(x_batch[bth] * mask, dim=0) / torch.sum(mask)
        # prototypes (B, Class, K, d)
        prototypes = self.proto_attention.multi_proto(x_batch, batch_mask, mu=proto_mu, node_types=node_type,
                                                      batch=batch, batch_num=batch_num)
        return prototypes, proto_mu


class ProtoAttention(torch.nn.Module):
    def __init__(self, num_proto, num_classes, iters=2, input_dim=1024,
                 hidden_dim=256, dim_proto=512, eps=1e-9, n_head=8, drop_rate=0.2):
        super().__init__()
        self.num_proto = num_proto
        self.iters = iters
        self.eps = eps
        self.dim_proto = dim_proto
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.n_head = n_head

        ### multi proto part
        self.proto_logsigma = nn.Parameter(torch.zeros(num_classes, 1, dim_proto))  # (Class, 1, d)
        init.xavier_uniform_(self.proto_logsigma)
        self.rand_mul = nn.Parameter(torch.randn(num_classes, self.num_proto, dim_proto))  # (Class, 1, d)
        # init.xavier_uniform_(self.proto_bias)
        self.proto_mlp = nn.Sequential(*[nn.Linear(input_dim, dim_proto),
                                         nn.LayerNorm(dim_proto),
                                         GELU(),
                                         nn.Dropout(drop_rate)])
        self.layers = nn.ModuleList(
            [MultiHeadCrossAttention(x_dim=input_dim, dim_proto=dim_proto, hidden_dim=hidden_dim, n_head=n_head,
                                     attn_pdrop=drop_rate, resid_pdrop=drop_rate)
             for _ in range(self.num_classes * self.iters)]
        )

    def multi_proto(self, x, batch_mask, batch, batch_num, mu, node_types):  # [(M_1,C), (M_2,C),...,(M_class,C)]
        mu = self.proto_mlp(mu)  # b,c,d
        mu = mu.view(batch_num, self.num_classes, 1, -1)  # b,c,1,d
        mu = mu.expand(-1, -1, self.num_proto, -1)  # b,c,n_p,d

        sigma = self.proto_logsigma.exp().expand(-1, self.num_proto, -1)  # 3, 8, 512
        all_proto = mu + sigma * self.rand_mul

        cnt = 0
        output = []
        for i in range(self.num_classes):
            proto = all_proto[:, i, ...]  # b cls num_p d
            for _ in range(self.iters):
                proto, _ = self.layers[cnt](proto, x, batch_mask=batch_mask)
                cnt += 1
            proto = proto.reshape([proto.shape[0], 1, proto.shape[1], proto.shape[2]])
            output.append(proto)  # [(b,1,k,d), (b,1,k,d), ..., (b,1,k,d)]
        output = torch.concatenate(output, dim=1)
        output = output.contiguous()
        return output  # (B, Class, K, d)


class ProtoFusion(torch.nn.Module):
    def __init__(self, x_dim=512, dim_proto=512, hidden_dim=512, n_head=8, fusion_layer=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiHeadCrossAttention(x_dim=x_dim, dim_proto=dim_proto, hidden_dim=hidden_dim, n_head=n_head)
             for _ in range(fusion_layer)]
        )

    def proto_conv(self, q_in, kv_in, batch_mask=None, get_attn=False):  # (Class*K,d), (m,d)
        att_scores = []
        for idx, encoder_layer in enumerate(self.layers):
            q_in, attn = encoder_layer(q_in, kv_in, batch_mask=batch_mask)
            if get_attn:
                att_scores.append(attn.detach().cpu().numpy())
        return q_in, att_scores


class MultiHeadCrossAttention(torch.nn.Module):
    def __init__(self, x_dim=512, dim_proto=512, hidden_dim=512, n_head=8, attn_pdrop=0.2, resid_pdrop=0.2):
        super().__init__()
        self.dim_proto = dim_proto
        self.n_head = n_head
        self.to_q = nn.Linear(dim_proto, dim_proto)
        self.to_k = nn.Linear(x_dim, dim_proto)
        self.to_v = nn.Linear(x_dim, dim_proto)

        self.mlp = nn.Sequential(
            nn.Linear(dim_proto, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, dim_proto)
        )

        self.mlp_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.att_norm = nn.LayerNorm(dim_proto)
        self.post_norm = nn.LayerNorm(dim_proto)

        self.out_proj = nn.Linear(dim_proto, dim_proto)

    def forward(self, q_in, kv_in, batch_mask=None):
        # self attention
        residual = q_in
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)
        q = self.to_q(q_in)

        q, attn = self.multi_head_attention(q, k, v, batch_mask)
        q = residual + self.att_norm(q)

        # feedforward
        residual = q
        q = self.mlp(q)
        q = self.mlp_drop(q)
        q = residual + self.post_norm(q)

        return q, attn

    def multi_head_attention(self, q, k, v, batch_mask=None):
        q = q.view(q.shape[0], q.shape[1], self.n_head, q.shape[2] // self.n_head)  # (B, T, nh,  hs)
        if len(k.shape) < 4:
            k = k.view(k.shape[0], k.shape[1], self.n_head, k.shape[2] // self.n_head)  # (B, T, nh,  hs)
            v = v.view(v.shape[0], v.shape[1], self.n_head, v.shape[2] // self.n_head)  # (B, T, nh,  hs)

        q = F.layer_norm(q, normalized_shape=(q.shape[-1],))
        k = F.layer_norm(k, normalized_shape=(k.shape[-1],))
        # v = F.layer_norm(v, normalized_shape=(v.shape[-1],))

        dots = torch.einsum('bkhd,bnhd->bkhn', q, k) * (1.0 / math.sqrt(k.size(-1)))  # (K,N)
        if batch_mask is not None:
            if batch_mask.shape[1] == q.shape[1]:
                dots = dots.masked_fill(batch_mask.unsqueeze(2).unsqueeze(3) == 0, mask_value)
            else:
                dots = dots.masked_fill(batch_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)
        attn = dots.softmax(dim=-1)  # (b,K,N)
        updates = torch.einsum('bnhd,bkhn->bkhd', v, attn)  # (K,N) @ (N,d) = (b,K,d)
        updates = updates.view(updates.shape[0], updates.shape[1],
                               updates.shape[2] * updates.shape[3]).contiguous()
        updates = self.out_proj(updates)
        updates = self.resid_drop(updates)

        return updates, attn
