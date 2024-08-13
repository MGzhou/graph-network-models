#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/10 17:15:49
@Desc : graphsage模型
'''
import random
import numpy as np
import torch
from torch import nn
from aggregator import MeanAggregator, GCNAggregator, PoolAggregator, LSTMAggregator

class SAGELayer(nn.Module):
    def __init__(self, self_dim, neig_dim, out_dim, aggregator_type="mean", sub_batch_size=64, dropout=0.0, concat=False, activation=True, bias=False, device="cpu"):
        """
        图聚合层

        Args:
            self_dim (int): 节点特征维度
            neig_dim (int): 邻居节点特征维度
            out_dim (int): 输出特征维度
            aggregator_type (str): 聚合类型, [mean, gcn, pool, lstm], pool 是 MaxPool. Defaults to "mean".
            sub_batch_size (int): 子聚合层聚合批次大小
            dropout (float): 失活概率
            concat (float): 是否进行自连接
            residual (bool): 是否进行残差连接. Defaults to Fasle.
            activation (bool): 最后是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
        """
        super(SAGELayer, self).__init__()

        self.aggregator_type = aggregator_type
        self.device = device

        # 子聚合层聚合批次大小
        self.sub_batch_size = sub_batch_size

        if aggregator_type == "mean":
            self.aggregator = MeanAggregator(self_dim, neig_dim, out_dim, dropout, concat, activation, bias)
        elif aggregator_type == "gcn":
            self.aggregator = GCNAggregator(self_dim, neig_dim, out_dim, dropout, activation, bias)
        elif aggregator_type == "pool":
            self.aggregator = PoolAggregator(self_dim, neig_dim, out_dim, dropout, concat, activation, bias)
        else:
            raise ValueError("aggregator_type must be mean, gcn, pool or lstm")
        
    def forward(self, nodes, neig_nodes, feats, h_neig_feats=None):
        """
        Args:
            nodes: 目标节点, (B,)
            neig_nodes: 邻居节点, (B, Ne)
            feats: 节点特征, (N, in_dim)
            h_neig_feats: 隐藏层节点特征, (Ne_{k-1}, hid_dim)
        """
        # 分批次
        batch_size = self.sub_batch_size
        outputs = []

        # 看后面可能需要将向量放进GPU
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i+batch_size]
            batch_neig_nodes = neig_nodes[i:i+batch_size]
            # 获取目标节点特征 # 如果device='cuda, 转换到GPU
            self_feats = feats[batch_nodes].to(self.device)  # 目标节点特征,  (B, self_dim)
            # 获取邻居节点特征
            a, b = batch_neig_nodes.shape  # (2 3)
            flatten_neig_nodes = torch.flatten(batch_neig_nodes)  # (6,)
            if h_neig_feats is not None:
                neighbor_feats = h_neig_feats[flatten_neig_nodes].view(a, b, -1)  # 邻居节点特征, (B, Ne, hid_dim) , Ne 是邻居节点数
            else:
                neighbor_feats = feats[flatten_neig_nodes].view(a, b, -1)  # 邻居节点特征, (B, Ne, hid_dim) , Ne 是邻居节点数
                neighbor_feats = neighbor_feats.to(self.device)
            
            # 聚合
            sub_output = self.aggregator(self_feats, neighbor_feats)
            outputs.append(sub_output)
        # 拼接
        output = torch.cat(outputs, dim=0)
        return output


"""问题
在第一层SAGEConvLayer中, 目标节点的向量已经变为了 out_dim 维度, 邻居节点向量还是 in_dim 维度, 

这样聚合之后, 邻居节点向量维度和目标节点向量维度不一致, 如何处理?

方式1: 在聚合之前使用线性变换将邻居节点向量变为 out_dim 维度, 这样聚合之后, 邻居节点向量维度和目标节点向量维度一致
"""

class GraphSAGE(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dim, 
        out_dim, 
        k=2,
        num_sample=8,
        agg="mean",
        sub_batch_size=64,
        dropout=0.6, 
        concat=False,
        activation=True, 
        bias=False,
        graph_type="edge",
        device="cpu"
    ):
        """
        GraphSAGE模型

        Args:
            in_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            out_dim (int): 输出维度，分类任务时是类别数。
            k (int): 聚合次数, 也是邻居节点阶数
            num_sample (int): 邻居节点采样数
            agg (str): 聚合方式, mean, gcn, pool, lstm
            dropout (float): 失活概率
            concat (bool): 是否将邻居节点特征和目标节点特征拼接, gcn 聚合器不生效
            activation (bool): 输出是否使用激活函数. Defaults to True.
            bias (bool): 是否使用偏置. Defaults to Fasle.
            graph_type (str): 图格式, dict, edge(src_nodes, dst_nodes)
            device (str): 模型运行设备. Defaults to "cpu".
        """
        super(GraphSAGE, self).__init__()
        hid_dim = hidden_dim
        sub_bs = sub_batch_size
        act = activation
        self.k = k
        self.num_sample = num_sample
        self.graph_type = graph_type
        # 聚合层
        self.sage_layers = nn.ModuleList()
        for i in range(self.k):
            if i == self.k - 1:
                neig_dim = in_dim
            else:
                neig_dim = hid_dim
            self.sage_layers.append(SAGELayer(in_dim, neig_dim, hid_dim, agg, sub_bs, dropout, concat, act, bias, device))
        
        # 分类层
        self.end_fc = nn.Linear(hidden_dim, out_dim)
        # 参数初始化
        nn.init.xavier_uniform_(self.end_fc.weight, gain=1.414)

    def forward(self, nodes, graph, feats):
        """
        实时采样进行聚合
        Args:
            nodes (tensor): 节点索引
            graph : 图数据
                if graph_type == "dict": {'id':[adj_list]}
                if graph_type == "edge": np.array([[src_nodes], [dst_nodes]])
            feats (tensor): 所有节点特征
        """
        # 采样
        if self.graph_type == "dict":
            neig_blocks = self.sample_neig(nodes, graph, self.k, self.num_sample)
        elif self.graph_type == "edge":
            neig_blocks = self.sample_neig_edge_graph(nodes, graph, self.k, self.num_sample)
        # 聚合
        # 采样是从左到右, 聚合是从右到左
        for i in range(self.k - 1, -1, -1):
            # 2, 1, 0
            sagelayer = self.sage_layers[i]
            if i == self.k-1:
                h = sagelayer(neig_blocks[i][0], neig_blocks[i][1], feats)
            else:
                h = sagelayer(neig_blocks[i][0], neig_blocks[i][1], feats, h)

        output = self.end_fc(h)  # (B, hid_dim) -> (B, out_dim)
        return output
    

    def sample_neig_edge_graph(self, nodes, graph, k, num_sample):
        """
        采样邻居节点

        后面可以考虑使用多线程加速

        Args:
            nodes (tensor): 节点索引
            graph (dict(list)): 图(边), arrar([v1,v2...], [u1,u2,...]), 边 v1->u1
            k (int): 采样层数, 也是聚合次数
            num_sample (int): 采样邻居节点数
        """
        neig_blocks = []
        current_nodes = nodes
        for i in range(k):
            neig_nodes = []
            for node in current_nodes:
                indices = np.where(graph[0] == node)[0]
                # 处理孤立节点，赋值自身
                neig_node = graph[1][indices] if len(indices) > 0 else np.array([node])
                if len(neig_node) >= num_sample:
                    # 无放回采样
                    neig_node = np.random.choice(neig_node, num_sample, replace=False)
                else:
                    # 有放回采样
                    neig_node2 = np.random.choice(neig_node, num_sample-len(neig_node), replace=True)
                    neig_node = np.concatenate((np.random.choice(neig_node, len(neig_node), replace=False), neig_node2))
                neig_nodes.append(neig_node)
            neig_blocks.append([torch.tensor(current_nodes), torch.tensor(np.array(neig_nodes))])
            # 获取当前层的主节点
            current_nodes = np.unique(np.concatenate(neig_nodes))
        # 转换中间层邻居节点索引  [[a, bs], [b, cs], [c, d]], 需要改变bs, cs的索引, 因为需要和每层返回的 h 对应
        if k > 1:
            for i in range(0, k-1):
                neig_node_k = neig_blocks[i][1]
                main_node_k_1 = neig_blocks[i+1][0]
                # 使用广播和高级索引获取索引
                indices = (main_node_k_1 == neig_node_k.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
                neig_blocks[i][1] = indices.view(neig_node_k.size())

        return neig_blocks  # [(main_nodes, neig_nodes), (main_nodes, neig_nodes), ...]


    def sample_neig(self, nodes, graph, k, num_sample):
        """
        采样邻居节点

        后面可以考虑使用多线程加速

        Args:
            nodes (tensor): 节点索引
            graph (dict(list)): 图邻接表, {node:[neighbor1, neighbor2, ...]}
            k (int): 采样层数, 也是聚合次数
            num_sample (int): 采样邻居节点数
        """
        neig_blocks = []  # 存储每一层的元组, (main_nodes, neig_nodes)
        current_nodes = nodes
        for i in range(k):
            neig_nodes = []
            for node in current_nodes:
                # 处理孤立节点，赋值自身
                neig_node = graph[node] if node in graph else [node]
                if len(neig_node) >= num_sample:
                    # 无放回采样
                    neig_node = np.random.choice(neig_node, num_sample, replace=False)
                else:
                    # 有放回采样
                    neig_node2 = np.random.choice(neig_node, num_sample-len(neig_node), replace=True)
                    neig_node = np.concatenate((np.random.choice(neig_node, len(neig_node), replace=False), neig_node2))
                neig_nodes.append(neig_node)
            # torch.int32
            neig_blocks.append([torch.tensor(current_nodes), torch.tensor(np.array(neig_nodes))])
            # 获取当前层的主节点
            current_nodes = np.unique(np.concatenate(neig_nodes))
        # 转换中间层邻居节点索引  [[a, bs], [b, cs], [c, d]], 需要改变bs, cs的索引, 因为需要和每层返回的 h 对应
        if k > 1:
            for i in range(0, k-1):
                neig_node_k = neig_blocks[i][1]
                main_node_k_1 = neig_blocks[i+1][0]
                # 使用广播和高级索引获取索引
                indices = (main_node_k_1 == neig_node_k.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
                neig_blocks[i][1] = indices.view(neig_node_k.size())

        return neig_blocks  # [(main_nodes, neig_nodes), (main_nodes, neig_nodes), ...]


if __name__=="__main__":
    # 测试 SAGEConvLayer
    in_dim = 8
    hidden_dim = 4
    out_dim = 2
    k = 2
    num_sample = 3
    agg = "pool"
    sub_batch_size = 2
    dropout = 0.0
    concat = True
    activation = True
    bias=False
    graph_type="edge"
    device = "cpu"

    adj_dict = {
        0: [1, 2, 3, 5],
        1: [0, 2, 3, 4],
        2: [0, 1, 3],
        3: [0, 2],
        4: [1, 5],
        5: [0, 4]
    }
    adj_edge = np.array([
        [0,0,0,0,1,1,1,1, 2,2,2, 3,3, 4,4, 5,5],
        [1,2,3,5,0,2,3,4, 0,1,3, 0,2, 1,5, 0,4]
    ])

    gs = GraphSAGE(in_dim, hidden_dim, out_dim, k, num_sample, agg, sub_batch_size, dropout, concat, activation, bias,graph_type, device)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    # 测试采样
    # neig_blocks = gs.sample_neig([0, 1], adj_dict, 2, 3)
    # print(neig_blocks)

    # 测试模型
    x = torch.tensor([[-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091],
        [-0.7121,  0.3037, -0.7773, -0.2515, -0.2223,  1.6871,  0.2284,  0.4676],
        [-0.6970, -1.1608,  0.6995,  0.1991,  0.8657,  0.2444, -0.6629,  0.8073],
        [ 1.1017, -0.1759, -2.2456, -1.4465,  0.0612, -0.6177, -0.7981, -0.1316],
        [ 1.8793, -0.0721,  0.1578, -0.7735,  0.1991,  0.0457,  0.1530, -0.4757],
        [-0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373,  1.5748, -0.6298]])
    x = x.to(device)
    train_data = torch.tensor([0,1,2], dtype=torch.long)
    h = gs(train_data.numpy(), adj_edge, x)
    print(h)







