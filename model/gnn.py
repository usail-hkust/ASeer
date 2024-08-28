import math
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, args, in_feat, nhid, dropout, layer, is_pred):
        super(SpGraphAttentionLayer, self).__init__()
        self.is_pred = is_pred
        # self.w_key = nn.Linear(in_feat, nhid, bias=True)
        # self.w_value = nn.Linear(in_feat, nhid, bias=True)
        # self.w_edge = nn.Linear(3+args.te_dim, nhid, bias=True)
        self.w_att = nn.Linear(4+3+args.te_dim-2*is_pred, nhid, bias=True)
        self.va = nn.Parameter(torch.zeros(1,nhid))
        nn.init.normal_(self.va.data)
        # self.w_out = nn.Linear(2+3+args.te_dim, nhid, bias=True)
        self.mlp_out = nn.Sequential(
            nn.Linear(2+3+args.te_dim, nhid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, nhid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, nhid, bias=True),
            nn.ReLU(inplace=True)
        )

        # self.leakyrelu = nn.LeakyReLU(0.2)
        # self.cosinesimilarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    def edge_attention(self, edges):
        # edge UDF
        # att_sim = torch.sum(torch.mul(edges.src['h_key'], edges.dst['h_key']),dim=-1)  # dot-product attention
        # att_sim = self.cosinesimilarity(edges.src['h_key'], edges.dst['h_key'])  # cosine attention
        # print(edges.dst['h_key'].shape)
        if(self.is_pred):
            xa = torch.cat([edges.src['h_x'], edges.data['h_edge']],dim=-1) 
        else:
            xa = torch.cat([edges.src['h_x'], edges.dst['h_x'], edges.data['h_edge']],dim=-1)
        att_sim = torch.sum(self.va*torch.tanh(self.w_att(xa)),dim=-1)
        return {'att_sim': att_sim}

    def message_func(self, edges):
        # message UDF
        return {'h_x': edges.src['h_x'], 'att_sim': edges.data['att_sim'], 'h_edge': edges.data['h_edge']}

    def reduce_func(self, nodes):
        # reduce UDF
        # print(nodes.mailbox['att_sim'].shape,nodes.mailbox['h_value'].shape) # (1, n_edge)
        alpha = F.softmax(nodes.mailbox['att_sim'], dim=1) # (# of nodes, # of neibors)
        alpha = alpha.unsqueeze(-1)
        # print(nodes.mailbox['h_value'].shape, nodes.mailbox['edge_feature'].shape)
        ### add edge features
        nodes_msgs = torch.cat([nodes.mailbox['h_x'], nodes.mailbox['h_edge']],dim=-1)
        h_att = torch.sum(alpha * nodes_msgs, dim=1)
        return {'h_att': h_att}

    def EDGE_TE(self, TE_Params, dt, dst_lane):
        # dt: (N_edge, 1)
        # dst_lane: (N_edge,)
        N_edge, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params  # (N, D)
        # print(TE_w.mean(), TE_b.mean())
        ret_TE_w = TE_w[dst_lane] # (N_edge, D)
        ret_TE_lam = TE_lam[dst_lane]
        # print(dst_lane.shape, dt.shape, ret_TE_w.shape)
        ind_sin = torch.sin(dt*ret_TE_w)
        ind_cos = torch.cos(dt*ret_TE_w)
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)

        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)
    
        lam = torch.exp(-torch.square(ret_TE_lam))
        # print(te_ind.shape, te_shared.shape, lam.shape)
        TE = (1-lam) * te_ind + lam * te_shared

        return TE
    
    def forward(self, TE_Params, X_msg, g):
        """
        :param X_key: X_key data of shape (num_nodes(N), in_features_1).
        :param X_value: X_value dasta of shape (num_nodes(N), in_features_2).
        :param g: sparse graph, edata(distance, delta_t, dst_period, dst_lane).
        :return: Output data of shape (num_nodes(N), out_features).
        """
        N, in_features = X_msg.size()
        # h_key = self.w_key(X_key)  # (B,N,out_features)
        # h_value = self.w_value(X_value)
        g.ndata['h_x'] = X_msg
        ### Edge TE ###
        # edge features: x_distance, x_edge_dt, x_reachability, x_dst_lane
        # delta_t = g.edata['feature'][...,1:2]/g.edata['feature'][...,2:3]
        delta_t = g.edata['feature'][...,1:2]
        dst_lane = g.edata['feature'][...,3].long()
        e_te = self.EDGE_TE(TE_Params, delta_t, dst_lane)
        g.edata['h_edge'] = torch.cat([g.edata['feature'][...,:3], e_te], dim=-1) # (n_edge, Fe)
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_att = g.ndata.pop('h_att') # (N,out_features)
        # print(h_att.shape)
        # h_conv = torch.relu(self.w_out(h_att))
        h_conv = self.mlp_out(h_att)
        return h_conv
        
class AGDN(nn.Module):
    def __init__(self, args, in_feat, nhid, is_pred=False, gathop=1, dropout=0):
        """sparse GAT."""
        super(AGDN, self).__init__()
        # print('# of gat layer:', gathop)
        self.nhid = nhid
        self.device = args.device
        self.dropout = nn.Dropout(dropout)
        self.gat_stacks = nn.ModuleList()
        for i in range(gathop):
            if(i > 0): in_feat = nhid 
            att_layer = SpGraphAttentionLayer(args, in_feat, nhid, dropout=dropout, layer=i, is_pred=is_pred)
            self.gat_stacks.append(att_layer)

    def forward(self, TE_Params, X_msg, adj): # no self-loop
        # lastx = torch.zeros((1, self.nhid)).to(self.device)
        for att_layer in self.gat_stacks:
            # out = self.dropout(att_layer(TE_Params, X_key, X_value, adj) + lastx)
            # X_key = out
            # X_value = out
            # lastx = out
            out = att_layer(TE_Params, X_msg, adj)
        return out
