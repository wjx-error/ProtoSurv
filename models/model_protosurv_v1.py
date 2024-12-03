import numpy as np
import torch
from torch.nn import Sequential as Seq
from torch.nn import LayerNorm, GELU
from torch_geometric.nn import GINConv, GENConv, DeepGCNLayer, global_mean_pool
from models.utils.model_utils import *
from models.utils.proto_utils_v1 import ProtoFusion, ProtoGNN
import random


#  hidden_dim=128
class LINKX_PROTO_oldv(torch.nn.Module):
    def __init__(self,
                 # general settings
                 input_dim=1024, dropout_rate=0.25,

                 # GNN settings
                 num_layers=4, edge_agg='spatial',
                 hidden_dim=128,

                 # proto settings
                 nr_types=6, num_proto=8,
                 proto_iters=2, proto_n_head=3,
                 fusion_layer_to_p=4, fusion_n_head=3,
                 dim_proto=768, proto_fc_hidden_dim=512,

                 # pre prototypes
                 pre_proto=False,
                 pre_proto_pth=None,

                 **kwargs
                 ):
        super(LINKX_PROTO_oldv, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1
        self.hidden_dim = hidden_dim
        self.max_input_len = 25_000
        self.dim_proto = dim_proto
        self.num_proto = num_proto
        self.nr_types = nr_types

        print('use pre_proto:', pre_proto)
        if pre_proto:
            print(f'model use pre prototypes from: {pre_proto_pth}')
        ### GNN part
        self.pre_fc = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      GELU(),
                                      nn.Dropout(dropout_rate)])
        self.total_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.total_layers.append(PatchGCN_module(hidden_dim, i, dropout_rate))
        self.post_fc = nn.Sequential(*[nn.Linear(hidden_dim * num_layers, dim_proto),
                                       nn.LayerNorm(dim_proto),
                                       GELU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(dim_proto, dim_proto)
                                       ])
        ### proto part
        self.ProtoGNN = ProtoGNN(feature_dim=input_dim, hidden_dim=proto_fc_hidden_dim, nr_types=nr_types,
                                 iters=proto_iters, dim_proto=dim_proto, num_proto=num_proto, n_head=proto_n_head,
                                 pre_proto=pre_proto, pre_proto_pth=pre_proto_pth
                                 )
        # fusion part
        self.proto_fusion_to_p = ProtoFusion(x_dim=dim_proto, dim_proto=dim_proto, hidden_dim=proto_fc_hidden_dim,
                                             n_head=fusion_n_head, fusion_layer=fusion_layer_to_p)
        ### head
        self.pred_norm = nn.LayerNorm(dim_proto, elementwise_affine=False)
        self.risk_prediction_layer = nn.Linear(dim_proto, 1, bias=False)

        # fc map for comp loss
        self.mlp_comp = nn.Sequential(*[nn.Linear(dim_proto, input_dim, bias=False),
                                        nn.LayerNorm(input_dim),
                                        GELU(),
                                        nn.Linear(input_dim, input_dim, bias=False)
                                        ])

    def forward(self, data):
        if self.training:
            max_input_len = self.max_input_len
        else:
            max_input_len = np.inf

        # get basic info
        edge_index = data.edge_index
        node_types = data.patch_classify_type  # n*3
        x = data.x
        batch = data.batch
        edge_attr = None
        batch_num = batch[-1] + 1
        node_types = node_types.unsqueeze(1)

        ### get prototypes
        # make batch
        seed = random.randint(0, int(1e6))
        x_batched, batch_mask = pad_batch(x, batch, max_input_len=max_input_len, seed=seed, get_mask=False)
        node_types_batched, _ = pad_batch(node_types, batch, max_input_len=max_input_len, seed=seed, get_mask=False)
        node_types_batched = node_types_batched.squeeze(2)
        # prototypes
        prototypes, _ = self.ProtoGNN.get_prototypes(x_batched, batch_mask, node_types_batched, batch, batch_num)
        prototypes = prototypes.reshape(batch_num, -1, self.dim_proto)  # b,c,k,d -> b,c*k,d
        # for comp loss
        prototypes_comp = self.mlp_comp(prototypes)  # (b,Class*K,d) for comp loss

        ### GNN
        x_ = self.pre_fc(data.x)
        x = self.total_layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], dim=-1)
        for layer in self.total_layers[1:]:
            x = layer(x, edge_index)
            x_ = torch.cat([x_, x], dim=-1)
        x = self.post_fc(x_)
        x_context_batched, batch_mask = pad_batch(x, batch, max_input_len=max_input_len, get_mask=False)

        ### cross attention fusion
        prototypes, att_scores = self.proto_fusion_to_p.proto_conv(prototypes, x_context_batched, batch_mask)

        h = torch.mean(prototypes, dim=1)

        # head
        h = self.pred_norm(h)
        S = self.risk_prediction_layer(h)
        if len(S.shape) == 1:
            S = S.unsqueeze()
        return None, S, None, None, prototypes_comp
