#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/30 18:10:27
@Desc :聚合器
'''

import torch
from torch import nn


class MeanAggregator(nn.Module):
    def __init__(self, self_dim, neig_dim, out_dim, dropout=0.0, concat=False, activation=True, bias=False):
        """
        Mean 图聚合层

        Math:
            math1: h_u = mean(h_{N(u)})
            math2: if concat: h = concat(h_v, h_u)  else: h = h_v + h_u
            math3: h = W * h + b

        Args:
            self_dim (int): 节点特征维度
            neig_dim (int): 邻居节点特征维度
            out_dim (int): 输出特征维度
            dropout (float): 失活概率
            concat (float): 是否进行自连接
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(MeanAggregator, self).__init__()
        self.self_dim = self_dim
        self.neig_dim = neig_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.concat = concat
        self.use_activation = activation
        self.use_bias = bias

        # dropout 失活函数
        if self.dropout > 0.0:
            self.drop = nn.Dropout(self.dropout)

        # 节点线性变换
        self.self_fc = nn.Linear(self_dim, out_dim, bias=False)
        self.neighbor_fc = nn.Linear(neig_dim, out_dim, bias=False)
        
        # 最后节点处理
        if self.concat:
            self.end_fc = nn.Linear(out_dim * 2, out_dim, bias=bias)
        else:
            self.end_fc = nn.Linear(out_dim, out_dim, bias=bias)
        
        # 激活函数
        if self.use_activation:
            self.relu = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化
        nn.init.xavier_uniform_(self.self_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.neighbor_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.end_fc.weight, gain=1.414)


    def forward(self, self_feats, neighbor_feats):
        """
        Args:
            self_feats: 目标节点特征, (B, self_dim)
            neighbor_feats: 邻居节点特征, (B, Ne, hid_dim), Ne 是邻居节点数
        """
        # dropout
        if self.dropout > 0.0:
            self_feats = self.drop(self_feats)
            neighbor_feats = self.drop(neighbor_feats)
        # 聚合邻居节点
        neighbor_feats = torch.mean(neighbor_feats, dim=1)  # (B, Ne, hid_dim) -> (B, hid_dim)

        # 处理目标节点和邻居节点
        self_feats = self.self_fc(self_feats)  # (B, self_dim) -> (B, out_dim)
        neighbor_feats = self.neighbor_fc(neighbor_feats)  # (B, hid_dim) -> (B, out_dim)

        if self.concat:
            self_feats = torch.cat([self_feats, neighbor_feats], dim=1)
        else:
            self_feats = self_feats + neighbor_feats
        
        # 最后经过一个全连接层
        output = self.end_fc(self_feats)  # (B, out_dim)

        # 激活函数
        if self.use_activation:
            output = self.relu(output)

        return output


class GCNAggregator(nn.Module):
    def __init__(self, self_dim, neig_dim, out_dim, dropout=0.0, activation=True, bias=False):
        """
        GCN 图聚合层

        Math:
            math1: h = concat(h_v, h_u)
            math2: h = W * h + b

        Args:
            self_dim (int): 节点特征维度
            neig_dim (int): 邻居节点特征维度
            out_dim (int): 输出特征维度
            dropout (float): 失活概率
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(GCNAggregator, self).__init__()

        self.dropout = dropout
        self.use_activation = activation
        
        # dropout 失活函数
        if self.dropout > 0.0:
            self.drop = nn.Dropout(dropout)

        # 目标节点线性变换
        self.self_fc = nn.Linear(self_dim, neig_dim, bias=False)
        
        # 最后节点变换
        self.end_fc = nn.Linear(neig_dim, out_dim, bias=bias)

        # 激活函数
        if self.use_activation:
            self.relu = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化
        nn.init.xavier_uniform_(self.self_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.end_fc.weight, gain=1.414)


    def forward(self, self_feats, neighbor_feats):
        """
        Args:
            self_feats: 目标节点特征, (B, self_dim)
            neighbor_feats: 邻居节点特征, (B, Ne, hid_dim), Ne 是邻居节点数
        """
        # dropout
        if self.dropout > 0.0:
            self_feats = self.drop(self_feats)
            neighbor_feats = self.drop(neighbor_feats)
        # 转为目标节点维度
        self_feats = self.self_fc(self_feats)  # (B, self_dim) -> (B, hid_dim)
        
        # 拼接目标节点与邻居节点
        neighbor_feats = torch.cat([self_feats.unsqueeze(1), neighbor_feats], dim=1)  # (B, Ne+1, hid_dim)
        # 聚合邻居节点
        neighbor_feats = torch.mean(neighbor_feats, dim=1)  # (B, Ne+1, hid_dim) -> (B, hid_dim)

        # 转换
        output = self.end_fc(neighbor_feats)  # (B, hid_dim) -> (B, out_dim)

        # 激活函数
        if self.use_activation:
            output = self.relu(output)

        return output

class PoolAggregator(nn.Module):
    def __init__(self, self_dim, neig_dim, out_dim, dropout=0.0, concat=False, activation=True, bias=False):
        """
        MaxPool 图聚合层

        Math:
            math1: h_u = maxpool(h_{N(u)})
            math2: if concat: h = concat(h_v, h_u)  else: h = h_v + h_u
            math3: h = W * h + b

        Args:
            self_dim (int): 节点特征维度
            neig_dim (int): 邻居节点特征维度
            out_dim (int): 输出特征维度
            dropout (float): 失活概率
            concat (float): 是否进行自连接
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(PoolAggregator, self).__init__()
        self.self_dim = self_dim
        self.neig_dim = neig_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.concat = concat
        self.use_activation = activation
        self.use_bias = bias

        # dropout 失活函数
        if self.dropout > 0.0:
            self.drop = nn.Dropout(self.dropout)

        # 最大池化之前做一次线性变换
        self.max_fc = nn.Linear(neig_dim, out_dim, bias=bias)

        # 节点线性变换
        self.self_fc = nn.Linear(self_dim, out_dim, bias=False)
        self.neighbor_fc = nn.Linear(out_dim, out_dim, bias=False)
        
        # 最后节点处理
        if self.concat:
            self.end_fc = nn.Linear(out_dim * 2, out_dim, bias=bias)
        else:
            self.end_fc = nn.Linear(out_dim, out_dim, bias=bias)
        
        # 激活函数
        if self.use_activation:
            self.relu = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化
        nn.init.xavier_uniform_(self.max_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.self_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.neighbor_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.end_fc.weight, gain=1.414)

    def forward(self, self_feats, neighbor_feats):
        """
        Args:
            self_feats: 目标节点特征, (B, self_dim)
            neighbor_feats: 邻居节点特征, (B, Ne, hid_dim), Ne 是邻居节点数
        """
        # dropout
        if self.dropout > 0.0:
            self_feats = self.drop(self_feats)
            neighbor_feats = self.drop(neighbor_feats)
        
        # 转换邻居节点维度
        neighbor_feats = self.max_fc(neighbor_feats)  # (B, Ne, hid_dim) -> (B, Ne, out_dim)

        # 最大池化
        neighbor_feats, _ = torch.max(neighbor_feats, dim=1)  # (B, Ne, out_dim) -> (B, out_dim)

        # 处理目标节点和邻居节点
        self_feats = self.self_fc(self_feats)  # (B, self_dim) -> (B, out_dim)
        neighbor_feats = self.neighbor_fc(neighbor_feats)  # (B, out_dim) -> (B, out_dim)

        if self.concat:
            self_feats = torch.cat([self_feats, neighbor_feats], dim=1)
        else:
            self_feats = self_feats + neighbor_feats

        # 最后经过一个全连接层
        output = self.end_fc(self_feats)  # (B, out_dim)

        # 激活函数
        if self.use_activation:
            output = self.relu(output)

        return output

class LSTMAggregator(nn.Module):
    def __init__(self, self_dim, neig_dim, out_dim, dropout=0.0, concat=False, activation=True, bias=False):
        """
        LSTM 图聚合层

        Math:
            math1: h_u = mean(h_{N(u)})
            math2: if concat: h = concat(h_v, h_u)  else: h = h_v + h_u
            math3: h = W * h + b

        Args:
            self_dim (int): 节点特征维度
            neig_dim (int): 邻居节点特征维度
            out_dim (int): 输出特征维度
            dropout (float): 失活概率
            concat (float): 是否进行自连接
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(LSTMAggregator, self).__init__()
        self.self_dim = self_dim
        self.neig_dim = neig_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.concat = concat
        self.use_activation = activation
        self.use_bias = bias

        # dropout 失活函数
        if self.dropout > 0.0:
            self.drop = nn.Dropout(self.dropout)

        # 节点线性变换
        self.self_fc = nn.Linear(self_dim, out_dim, bias=False)
        self.neighbor_fc = nn.Linear(neig_dim, out_dim, bias=False)
        
        # 最后节点处理
        if self.concat:
            self.end_fc = nn.Linear(out_dim * 2, out_dim, bias=bias)
        else:
            self.end_fc = nn.Linear(out_dim, out_dim, bias=bias)
        
        # 激活函数
        if self.use_activation:
            self.relu = nn.ReLU()


    def forward(self, self_feats, neighbor_feats):
        """
        Args:
            self_feats: 目标节点特征, (B, self_dim)
            neighbor_feats: 邻居节点特征, (B, Ne, hid_dim), Ne 是邻居节点数
        """
        # dropout
        if self.dropout > 0.0:
            self_feats = self.drop(self_feats)
            neighbor_feats = self.drop(neighbor_feats)
        # 聚合邻居节点
        neighbor_feats = torch.mean(neighbor_feats, dim=1)  # (B, Ne, hid_dim) -> (B, hid_dim)

        # 处理目标节点和邻居节点
        self_feats = self.self_fc(self_feats)  # (B, self_dim) -> (B, out_dim)
        neighbor_feats = self.neighbor_fc(neighbor_feats)  # (B, hid_dim) -> (B, out_dim)

        if self.concat:
            self_feats = torch.cat([self_feats, neighbor_feats], dim=1)
        else:
            self_feats = self_feats + neighbor_feats

        # 最后经过一个全连接层
        output = self.end_fc(self_feats)  # (B, out_dim)

        # 激活函数
        if self.use_activation:
            output = self.relu(output)

        return output















