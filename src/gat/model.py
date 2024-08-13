#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/10 17:15:49
@Desc : gat模型
'''
import torch
from torch import nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, residual=False, activation=True, bias=False):
        """
        图注意力层, paper: <https://arxiv.org/pdf/1710.10903.pdf>

        Math:
            math1 注意力层消息传播公式
            $$ h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)} $$

            math2 注意力系数
            $$ \alpha_{ij}^{l} = \mathrm{softmax_i} (e_{ij}^{l}) $$

            math3 相似系数
            $$ e_{ij}^{l} = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right) $$

        Args:
            in_dim (int): 输入特征维度
            out_dim (int): 输出特征维度
            dropout (float): 失活概率
            alpha (float): LeakyReLU 的负斜率
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 最后是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        self.use_activation = activation
        self.use_residual = residual
        self.use_bias = bias

        # 训练权重定义和初始化
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_att_left = nn.Parameter(torch.empty(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.W_att_left.data, gain=1.414)

        self.W_att_right = nn.Parameter(torch.empty(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.W_att_right.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
            nn.init.constant_(self.bias, 0)
        
        if self.use_residual and in_dim != out_dim:
            self.W_residual = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
            nn.init.xavier_uniform_(self.W_residual.data, gain=1.414)

        # 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # dropout 失活函数
        self.attn_drop = nn.Dropout(dropout)


    def forward(self, h, adj):
        # 计算 math1 后半部分
        Wh = torch.mm(h, self.W)  # W=(N, in_dim), W=(in_dim, out_dim)  --> Wh=(N, out_dim)  N为节点数量

        # 计算 math3, 这里和论文有出入
        # 这里不是使用输入h，而是使用经线性转换的h
        Wh1 = torch.matmul(Wh, self.W_att_left)  # (N, out_dim)x(out_dim,1) --> (N,1)
        Wh2 = torch.matmul(Wh, self.W_att_right) # (N, out_dim)x(out_dim,1) --> (N,1)
        e = Wh1 + Wh2.T  # (N, N)
        e = self.leakyrelu(e)
        # 计算 math2
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 非邻居节点注意力系数置零
        attention = F.softmax(attention, dim=1)
        attention = self.attn_drop(attention)  # (N, N)
        # 计算 math1
        h_output = torch.matmul(attention, Wh)  # (N,N)x(N,out_dim) --> (N,out_dim)
        
        if self.use_residual:
            if self.in_dim != self.out_dim:
                h = torch.mm(h, self.W_residual)
            h_output += h
        
        if self.use_bias:
            h_output += self.bias

        if self.use_activation:
            h_output = F.elu(h_output)

        return h_output

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.6, alpha=0.2, residual=False, bias=False):
        """
        GAT模型，双层GAT网络

        Args:
            in_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            out_dim (int): 输出维度，分类任务时是类别数。
            num_heads (int): 注意力头数
            dropout (float): 失活概率
            alpha (float): LeakyReLU 的负斜率
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 输出是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(GAT, self).__init__()
        # 多头注意力
        self.attentions = [GraphAttentionLayer(in_dim, hidden_dim, dropout=dropout, alpha=alpha, residual=residual, activation=True, bias=bias) for _ in range(num_heads)]
        # 添加子模块到模型模块，使 PyTorch 能够识别和管理这些子模块。
        for i, sub_attention in enumerate(self.attentions):
            self.add_module('attention_head_{}'.format(i), sub_attention)
        
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, out_dim, dropout=dropout, alpha=alpha, residual=residual, activation=False, bias=bias)

        # 失活函数
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.feat_drop(x)
        # 合并多头注意力输出
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.feat_drop(x)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)  # ln(softmax(x))
        # return x


if __name__=="__main__":
    # 测试 GraphAttentionLayer
    
    in_dim = 16
    hidden_dim = 8
    out_dim = 3
    dropout = 0.1
    alpha = 0.2
    num_heads = 2

    x = torch.rand((5, 16))
    adj = [
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ]
    adj = torch.FloatTensor(adj)
    gatlayer = GraphAttentionLayer(in_dim, hidden_dim, dropout, alpha)

    r = gatlayer(x, adj)
    print(r.shape)  # torch.Size([5, 8])

    # 测试 GAT
    gat = GAT(in_dim, hidden_dim, out_dim, num_heads, dropout, alpha)

    r = gat(x, adj)

    print(r.shape)
    print(r)
