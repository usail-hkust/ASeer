import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gnn import AGDN


class SemiAR_Decoder(nn.Module):
    def __init__(self, args, in_feat):
        """PeriodDecoder"""
        super(SemiAR_Decoder, self).__init__()
        self.n_output = args.n_output
        self.device = args.device
        self.update_GRU = nn.GRUCell(in_feat, args.hid_dim, bias=True)
        self.MLP_period = nn.Sequential(
            nn.Linear(in_feat + 2*args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.n_output, bias=True),
            nn.Sigmoid()
        )

        self.MLP_flow = nn.Sequential(
            nn.Linear(in_feat + 2*args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.n_output, bias=True),
        )

    def Decoder_IndividualTE(self, TE_Params, dt_onestep):
        dt = dt_onestep.unsqueeze(dim=-2) # (N, 1, 1)
        N, L, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params  # (N, D)
        ind_sin = torch.sin(dt*TE_w.repeat(1,L,1))
        ind_cos = torch.cos(dt*TE_w.repeat(1,L,1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)
        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)
        # TE = torch.sigmoid(TE_lam) * te_ind + (1-torch.sigmoid(TE_lam)) * te_shared
        lam = torch.exp(-torch.square(TE_lam))
        TE = (1-lam) * te_ind + lam * te_shared
        return TE.squeeze(dim=-2)

    def forward(self, TE_Params, h_t_int, dt_init, X, Y, is_test):
        """_summary_

        Parameters
        ----------
        h_t_int
            shape (N, hiddim)
        dt_pred
            shape (N, 1)
        Y
            shape (N, Ly, F)

        Returns
        -------
            shape (N, Ly)
        """
        N, Ly, _ = Y.shape
        Y_period = Y[...,0]
        Y_dt = Y[...,2]
        n_iter = math.ceil(Ly/self.n_output)
        period_outputs = []
        flow_outputs = []
        flow_test_outputs = []

        dt_pred = dt_init.clone()
        elasped_time = dt_init.clone()
        ht_period = h_t_int.clone()
        for k in range(n_iter):
            ### predict periods ###
            ht_period = self.update_GRU(torch.cat([elasped_time, self.Decoder_IndividualTE(TE_Params, elasped_time)],dim=-1), ht_period)
            input_p = torch.cat([dt_pred, self.Decoder_IndividualTE(TE_Params, dt_pred), h_t_int, ht_period],dim=-1)
            periods = self.MLP_period(input_p) # (N, self.n_output)
            period_outputs.append(periods)

            ### semi-autoregressively update beginning dt
            dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True) # (N, 1)
            elasped_time = torch.sum(periods, dim=-1, keepdim=True) # (N, 1)

        elasped_time_truth = dt_init.clone()
        ht_flow = h_t_int.clone()
        for k in range(n_iter):
            ### predict flows ###
            dt_truth = Y_dt[:,k*self.n_output:k*self.n_output+1]
            ht_flow = self.update_GRU(torch.cat([elasped_time_truth, self.Decoder_IndividualTE(TE_Params, elasped_time_truth)],dim=-1), ht_flow)
            input_f = torch.cat([dt_truth, self.Decoder_IndividualTE(TE_Params, dt_truth), h_t_int, ht_flow],dim=-1) # (N, hiddim)
            flows = self.MLP_flow(input_f) # (N, self.n_output, 2)
            flow_outputs.append(flows)

            ### semi-autoregressively update beginning dt
            elasped_time_truth = torch.sum(Y_period[:,k*self.n_output:(k+1)*self.n_output], dim=-1, keepdim=True)

        if(is_test):
            period_outputs = []
            dt_pred = dt_init.clone()
            elasped_time = dt_init.clone()
            ht = h_t_int.clone()
            k = 1
            while(True):
                ht = self.update_GRU(torch.cat([elasped_time, self.Decoder_IndividualTE(TE_Params, elasped_time)],dim=-1), ht)
                input = torch.cat([dt_pred, self.Decoder_IndividualTE(TE_Params, dt_pred), h_t_int, ht],dim=-1)
                periods = self.MLP_period(input)
                periods = torch.clip(periods, 0.05, 1)
                period_outputs.append(periods)

                flows_test = self.MLP_flow(input)
                flow_test_outputs.append(flows_test)

                dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True) # (N, 1)
                elasped_time = torch.sum(periods, dim=-1, keepdim=True) # (N, 1)
                
                if((k >= n_iter) and (300*dt_pred.min().item() > 7200)):
                    break
                k += 1

            flow_test_outputs = torch.cat(flow_test_outputs, dim=-1)
            
        period_outputs = torch.cat(period_outputs, dim=-1) # (N, n_iter*n_output)
        flow_outputs = torch.cat(flow_outputs, dim=-1) # (N, n_iter*n_output)
            
        return period_outputs, flow_outputs, flow_test_outputs


class SemiAR_Decoder_MLP(nn.Module):
    def __init__(self, args, in_feat):
        """PeriodDecoder"""
        super(SemiAR_Decoder_MLP, self).__init__()
        self.n_output = args.n_output
        self.device = args.device
        self.MLP_period = nn.Sequential(
            nn.Linear(in_feat + args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.n_output, bias=True),
            nn.Sigmoid()
        )

        self.MLP_flow = nn.Sequential(
            nn.Linear(in_feat + args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.n_output, bias=True),
        )

    def Decoder_IndividualTE(self, TE_Params, dt_onestep):
        dt = dt_onestep.unsqueeze(dim=-2) # (N, 1, 1)
        N, L, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params  # (N, D)
        ind_sin = torch.sin(dt*TE_w.repeat(1,L,1))
        ind_cos = torch.cos(dt*TE_w.repeat(1,L,1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)
        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)
        # TE = torch.sigmoid(TE_lam) * te_ind + (1-torch.sigmoid(TE_lam)) * te_shared
        lam = torch.exp(-torch.square(TE_lam))
        TE = (1-lam) * te_ind + lam * te_shared
        return TE.squeeze(dim=-2)

    def forward(self, TE_Params, h_t_int, dt_init, X, Y, is_test):
        """_summary_

        Parameters
        ----------
        h_t_int
            shape (N, hiddim)
        dt_pred
            shape (N, 1)
        Y
            shape (N, Ly, F)

        Returns
        -------
            shape (N, Ly)
        """
        N, Ly, _ = Y.shape
        Y_period = Y[...,0]
        Y_dt = Y[...,2]
        n_iter = math.ceil(Ly/self.n_output)
        period_outputs = []
        flow_outputs = []
        flow_test_outputs = []

        dt_pred = dt_init.clone()
        for k in range(n_iter):
            ### predict periods ###
            input_p = torch.cat([dt_pred, self.Decoder_IndividualTE(TE_Params, dt_pred), h_t_int],dim=-1)
            periods = self.MLP_period(input_p) # (N, self.n_output, 2)
            period_outputs.append(periods)

            ### predict flows ###
            dt_truth = Y_dt[:,k*self.n_output:k*self.n_output+1]
            input_f = torch.cat([dt_truth, self.Decoder_IndividualTE(TE_Params, dt_truth), h_t_int],dim=-1) # (N, hiddim)
            flows = self.MLP_flow(input_f) # (N, self.n_output, 2)
            flow_outputs.append(flows)

            ### semi-autoregressively update beginning dt
            dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True) # (N, 1)

        if(is_test):
            period_outputs = []
            dt_pred = dt_init.clone()
            k = 1
            while(True):
                input = torch.cat([dt_pred, self.Decoder_IndividualTE(TE_Params, dt_pred), h_t_int],dim=-1)
                periods = self.MLP_period(input)
                periods = torch.clip(periods, 0.05, 1)
                period_outputs.append(periods)
                flows_test = self.MLP_flow(input)
                flow_test_outputs.append(flows_test)

                dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True) # (N, 1)
                
                if((k >= n_iter) and (300*dt_pred.min().item() > 7200)):
                    break
                k += 1

            flow_test_outputs = torch.cat(flow_test_outputs, dim=-1)
            
        period_outputs = torch.cat(period_outputs, dim=-1) # (N, n_iter*n_output)
        flow_outputs = torch.cat(flow_outputs, dim=-1) # (N, n_iter*n_output)
            
        return period_outputs, flow_outputs, flow_test_outputs



class ASeer(nn.Module):
    def __init__(self, args):
        super(ASeer, self).__init__()
        self.device = args.device
        self.dropout = nn.Dropout(args.dropout)
        self.hid_dim = args.hid_dim

        # Seq delat_t TE
        self.sharedTE_w = nn.Linear(1, args.te_dim//2, bias=False)
        self.TE_w = nn.Parameter(torch.zeros(args.N, 1, args.te_dim//2))
        nn.init.normal_(self.TE_w.data)

        # GNN delat_t TE
        self.sharedTE_edge_w = nn.Linear(1, args.te_dim//2, bias=False)
        self.TE_edge_w = nn.Parameter(torch.zeros(args.N, args.te_dim//2))
        nn.init.normal_(self.TE_edge_w.data)
        
        if(args.te == "combine"):
            self.TE_lam = nn.Parameter(torch.zeros(args.N, 1, 1)+1e-6)
            self.TE_edge_lam = nn.Parameter(torch.zeros(args.N, 1)+1e-6)
        elif(args.te == "share"):
            self.TE_lam = torch.zeros(args.N, 1, 1).to(self.device)
            self.TE_edge_lam = torch.zeros(args.N, 1).to(self.device)
        elif(args.te == "ind"):
            self.TE_lam = torch.ones(args.N, 1, 1).to(self.device)*1e9
            self.TE_lam = torch.ones(args.N, 1).to(self.device)*1e9
        else:
            print("te input error")

        input_dim = 3 + args.te_dim + self.hid_dim
        self.Filter_Generators = nn.Sequential(
                nn.Linear(input_dim, args.hid_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(args.hid_dim, args.hid_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(args.hid_dim, input_dim*args.hid_dim, bias=True))
        self.T_bias = nn.Parameter(torch.zeros(1, args.hid_dim))
        nn.init.normal_(self.T_bias.data)
        # self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.AGDN_EN = AGDN(args, in_feat=2, nhid=args.hid_dim, is_pred=False)
        self.AGDN_REST = AGDN(args, in_feat=2, nhid=args.hid_dim, is_pred=True)

        if(args.decoder == "SEU"):
            print("Decoder: SEU")
            self.decoder = SemiAR_Decoder(args, input_dim-2-self.hid_dim)
        elif(args.decoder == "MLP"):
            print("Decoder: MLP")
            self.decoder = SemiAR_Decoder_MLP(args, input_dim-2-self.hid_dim)

    def IndividualTE(self, dt):
        # dt: (N, L, 1)
        N, L, _ = dt.shape
        ind_sin = torch.sin(dt*self.TE_w.repeat(1,L,1))
        ind_cos = torch.cos(dt*self.TE_w.repeat(1,L,1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)
        shared_sin = torch.sin(self.sharedTE_w(dt))
        shared_cos = torch.cos(self.sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)
        # TE = torch.sigmoid(self.TE_lam) * te_ind + (1-torch.sigmoid(self.TE_lam)) * te_shared
        lam = torch.exp(-torch.square(self.TE_lam))
        TE = (1-lam) * te_ind + lam * te_shared
        return TE

    def TTCN(self, X_int, mask_X):
        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*hid_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*hid_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.hid_dim, -1) # (N, Lx, hid_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.hid_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, hid_dim)
        # print(mask_X.shape, Filter_seqnorm.shape, ttcn_out.shape)
        h_t = torch.relu(ttcn_out + self.T_bias) # (N, hid_dim)
        # h_t = self.leakyrelu(ttcn_out + self.T_bias) # (N, hid_dim)
        # h_t = torch.tanh(ttcn_out + self.T_bias) # (N, hid_dim)
        return h_t


    def forward(self, X, mask_X, Y, adjs, is_test=False):
        # features of X: period, unit_flow, delta_t
        # old features of X: st_time, ed_time, mid_time, period, flow, unit_flow, delta_t
        # mask_X: (1, N, Lx)
        X = X.squeeze(dim=0) # (N, Lx, Fx)
        Y = Y.squeeze(dim=0)
        mask_X = mask_X.squeeze(dim=0).unsqueeze(dim=-1) # (N, Lx, 1)
        N, Lx, Fx = X.shape
        adj_X, adj_Y = adjs
        TE_Params = (self.TE_w, self.sharedTE_w, self.TE_lam)
        TE_edge_Params = (self.TE_edge_w, self.sharedTE_edge_w, self.TE_edge_lam)
        
        """ Encoder """
        X_msg = X[...,:2].reshape(N*Lx, -1) # (N*Lx, F)
        hx_msg = self.AGDN_EN(TE_edge_Params, X_msg, adj_X)
        hx_msg = hx_msg.reshape(N, Lx, -1) # (N, Lx, hid_dim)

        ### TTCN ###
        te = self.IndividualTE(X[...,-1:])
        X_int = torch.cat([X, te, hx_msg],dim=-1) # (N, Lx, F_in)
        h_t = self.TTCN(X_int, mask_X) # (N, hid_dim)

        """ Decoder """
        ### GAT_REST ###
        Y_msg = torch.cat([X_msg, torch.zeros_like(Y[:,:1,:2].reshape(N*1, -1))], axis=0) # (N*Lx+N, 1)
        hy_msg = self.AGDN_REST(TE_edge_Params, Y_msg, adj_Y)
        hy_msg = hy_msg[len(X_msg):].reshape(N, -1) # (N, hid_dim)

        ### Semi-autoregressive prediction ###
        h_t_int = h_t + hy_msg
        dt_start = Y[:,:1,2] # (N, 1)
        period_outputs, flow_outputs, flow_test_outputs = self.decoder(TE_Params, h_t_int, dt_start, X, Y, is_test) # (N, Ly)
        
        return period_outputs, flow_outputs, flow_test_outputs
