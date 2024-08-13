#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:07:10
@Desc :模型文件
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        图卷积网络层, paper:<https://arxiv.org/abs/1609.02907>
        
        GCN卷积公式
        f(H) = D^-1/2 A D^-1/2 * H * W

        adjacency = D^-1/2 A D^-1/2 已经经过归一化，标准化的拉普拉斯矩阵
        
        Args:
            input_dim (int): 节点输入特征的维度
            output_dim (int): 输出特征维度
            use_bias (bool): 是否使用偏置
        """
        super(GraphConvolutionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        #定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    # 这里才是声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
    
        Args: 
            input_feature (torch.Tensor): 输入特征
            adjacency (torch.sparse.FloatTensor): 邻接矩阵
        """
        support = torch.mm(input_feature, self.weight)  # 矩阵相乘, m是matrix缩写
        # 稀疏矩阵相乘。如果adjacency不是稀疏矩阵，则要更改为torch.mm
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias  # bias 偏置
        return output
    
    # 打印类实例的信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class GCN(nn.Module):
    def __init__(self, config):
        """图卷积神经网络模型

        Args:
            config (dict): 模型配置
        """
        super(GCN, self).__init__()
        self.config = config
        self.input_gcn_cov = GraphConvolutionLayer(config['input_dim'], config['hidden_dim'])
        # 在for循环中添加子模型，需要手动添加设备
        self.hidden_gcn_cov_layers = []
        for i in range(config['layers'] - 2):
            self.hidden_gcn_cov_layers.append(GraphConvolutionLayer(config['hidden_dim'], config['hidden_dim']).to(config['device']))
        self.output_gcn_cov = GraphConvolutionLayer(config['hidden_dim'], config['output_dim'])
    
    def forward(self, feature, adjacency):
        """前向传播

        Args:
            feature (torch.tensor): 节点特征
            adjacency (torch.sparse.tensor): 拉普拉斯邻居矩阵
            
        Returns:
            torch.tensor: 输出预测概率矩阵
        """
        h = F.relu(self.input_gcn_cov(feature, adjacency))  # cora (N,1433)->(N,16)
        for gcn_cov in self.hidden_gcn_cov_layers:
            h = F.relu(gcn_cov(h, adjacency))
        logits = self.output_gcn_cov(h, adjacency)  # cora (N,16)->(N,7), 没有经过 softmax 归一化
        return logits
