import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hid, dropout=0, act='relu'):
        super().__init__()
        self.dropout = dropout
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        if act == 'relu':
            self.f_act = nn.ReLU
        else:
            raise NotImplementedError
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hid, bias=True),
            self.f_act(),
            nn.Linear(self.dim_hid, self.dim_hid, bias=True)
        )

    def forward(self, feat_in):
        assert len(feat_in.shape) == 2 or len(feat_in.shape) == 3
        return self.mlp(feat_in)


class MLP_COMM(nn.Module):
    def __init__(self, dim_in, dim_hid, nagents, dropout=0, act='relu'):
        super().__init__()
        self.dropout = dropout
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.nagents = nagents
        if act == 'relu':
            self.f_act = nn.ReLU
        else:
            raise NotImplementedError
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in * self.nagents, self.dim_hid, bias=True),
            self.f_act()
        )

    def forward(self, feat_in, mask):
        if len(feat_in.shape) == 3:
            feat_ax = feat_in[:, np.newaxis, ...]
        else:
            feat_ax = feat_in
        d1, d2 = feat_in.shape[:2]
        feat_ma = (mask[..., np.newaxis].float() * feat_ax).reshape(d1, d2, -1)
        feat_trans = self.mlp(feat_ma)
        return feat_trans
        

class GNN(nn.Module):
    def __init__(self, dim_in, dim_hid, dropout=0, act='relu'):
        super().__init__()
        self.dropout = dropout
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        if act == 'relu':
            self.f_act = nn.ReLU
        else:
            raise NotImplementedError
    
    def forward(self, feat_in, mask):
        raise NotImplementedError

    def _aggr_sum(self, feat, mask):
        if len(feat.shape) == 4:
            feat_aggr = (mask[..., np.newaxis].float() * feat).sum(axis=-2)
        else:
            feat_aggr = torch.matmul(mask.float(), feat)
        return feat_aggr


class GIN(GNN):
    def __init__(self, dim_in, dim_hid, dropout=0, act='relu'):
        super().__init__(dim_in, dim_hid, dropout=dropout, act=act)
        self.mlp_pre = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hid, bias=True),
            self.f_act(),
            nn.Linear(self.dim_hid, self.dim_hid, bias=True)
        )
        self.mlp_post = nn.Sequential(
            nn.Linear(self.dim_hid, self.dim_hid, bias=True),
            self.f_act(),
            nn.Linear(self.dim_hid, self.dim_hid, bias=True),
        )
        self.f_dropout = nn.Dropout(p=self.dropout)
        
    def forward(self, feat_in, mask):

        feat_pre_aggr = self.mlp_pre(feat_in)
        feat_aggr = self._aggr_sum(feat_pre_aggr, mask)
        feat_post_aggr = self.mlp_post(feat_aggr)
        return feat_post_aggr      


class SAGE(GNN):
    def __init__(self, dim_in, dim_hid, dropout=0, act='relu'):
        super().__init__(dim_in, dim_hid, dropout=dropout, act=act)
        self.mlp_post = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hid, bias=True),
            self.f_act()
        )
        self.f_dropout = nn.Dropout(p=self.dropout)
    
    def forward(self, feat_in, mask):
        feat_aggr_sum = self._aggr_sum(feat_in, mask)
        feat_aggr = feat_aggr_sum / (mask.sum(axis=-2)[..., np.newaxis] + 1e-7)
        feat_trans = self.mlp_post(feat_aggr)
        return feat_trans


class HyperNetPermEquiv(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hid, n_hid=(0, 0), pool=None, act='relu'):
     
        super().__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.dim_hid = dim_hid
        self.n_hid_split, self.n_hid_aggr = n_hid
        self.pool = pool
        self.lin_layers_split = []
        if self.pool is None:
            if self.n_hid_split == 0:
                self.lin_layers_split.append(nn.Linear(self.dim_in, self.dim_out))   
            else:
                self.lin_layers_split.append(nn.Linear(self.dim_in, self.dim_hid))
                for i in range(self.n_hid_split - 1):
                    self.lin_layers_split.append(nn.Linear(self.dim_hid, self.dim_hid))
                self.lin_layers_split.append(nn.Linear(self.dim_hid, self.dim_out))
        else:
            self.lin_layers_split.append(nn.Linear(self.dim_in, self.dim_hid))
            for i in range(self.n_hid_split):
                self.lin_layers_split.append(nn.Linear(self.dim_hid, self.dim_hid))
            self.lin_layers_aggr = []            
            for i in range(self.n_hid_aggr):
                self.lin_layers_aggr.append(nn.Linear(self.dim_hid, self.dim_hid))
            self.lin_layers_aggr.append(nn.Linear(self.dim_hid, self.dim_out))            
            self.lin_layers_aggr = nn.ModuleList(self.lin_layers_aggr)
        self.lin_layers_split = nn.ModuleList(self.lin_layers_split)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'elu':
            self.act = nn.ELU()
        else:
            raise NotImplementedError


    def _aggr(self, feat):
        if self.pool == 'sage':
            feat = feat.mean(axis=2)
        elif self.pool == 'gin':
            feat = feat.sum(axis=2)
        else:
            raise NotImplementedError
        return feat

    def forward(self, feat_in):

        feat = feat_in
        for fl in self.lin_layers_split[:-1]:
            feat = fl(feat)
            feat = self.act(feat)
        feat = self.lin_layers_split[-1](feat)
        if self.pool is not None:
            assert len(self.lin_layers_aggr) > 0
            feat = self.act(feat)     
            feat = self._aggr(feat)
            for fl in self.lin_layers_aggr[:-1]:
                feat = fl(feat)
                feat = self.act(feat)
            feat = self.lin_layers_aggr[-1](feat)
        return feat
