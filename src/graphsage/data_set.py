#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:06:40
@Desc :读取和处理数据集的类文件
'''

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class GraphData(object):
    def __init__(self, config, rebuild=False, logger_info=print):
        """数据处理, 加载.

        处理之后的数据包括如下几部分(cora为例):
            * x(tensor): 节点的特征 (2708,1433)
            * y(tensor): 节点的标签, 总共包括7个类别 (2078,)
            * adjacency(np.ndarray):   二维边矩阵 (10848)      
            * train_ids(list):   训练集索引 (1624)
            * val_ids(list):     验证集索引 (542)
            * test_ids(list):    测试集索引 (542)

        Args:
            config (dict): 配置字典
            rebuild (boolean): 是否需要重构数据集, 默认读取存在的缓存。当设为True时, 就算存在缓存数据也会重建数据
            logger_info (function): 日志打印函数
        """
        self.rebuild = rebuild
        self.device = config['device']
        self.use_semi_supervised = config['use_semi_supervised']
        self.logger_info = logger_info  # 日志函数
        ## 处理数据集路径
        self.data_root = config['data_path']
        # 公开数据集
        if self.data_root in ['cora', 'citeseer', 'pubmed']:
            # 文件名称
            self.filenames = ["ind.{}.{}".format(self.data_root, name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
            # 名称转路径
            self.data_root = os.path.join(self.get_project_path(), 'data', self.data_root)
            # 缓存数据路径
            if self.use_semi_supervised:
                self.save_file = os.path.join(self.data_root, "graphsage_processed_data_semi.pkl")
            else:
                self.save_file = os.path.join(self.data_root, "graphsage_processed_data_sup.pkl")
        elif not os.path.exists(self.data_root):
            raise ValueError("data_root {} does not exists.".format(self.data_root))
        else:
            # 自定义数据集
            is_file = os.path.isfile(self.data_root)
            if is_file:
                self.rebuild = False
                self.save_file = self.data_root
            else:
                raise ValueError("data_root {} does not a file.".format(self.data_root))

    def get_data(self):
        """
        获取输入模型的数据, 包括节点特征、标签、训练集、验证集、测试集, 二维边矩阵
        注, graphsage模型在处理数据时均不将数据转移到GPU
        Return: 
        """
        dataset = self.pre_process()
        self.print_data_info(dataset)               # 打印数据集信息
        tensor_x = torch.tensor(dataset['x'])       # 转为torch.tensor, cora=(2708,1433)
        tensor_y = torch.tensor(dataset['y'])       # cora=(2708,)  .to(self.device)
        _, input_dim = tensor_x.shape               # 节点数量和维度, 
        class_num = tensor_y.max().item() + 1       # 类别数量
        self.input_dim, self.class_num = input_dim, class_num
        # 转换数据格式
        val_ids = torch.tensor(dataset["val_ids"], dtype=torch.long)
        test_ids = torch.tensor(dataset["test_ids"], dtype=torch.long)
        train_ids = torch.tensor(dataset["train_ids"], dtype=torch.long)
        adjacency = dataset['adjacency']
        return tensor_x, tensor_y, train_ids, val_ids, test_ids, adjacency

    def pre_process(self):
        """
        处理数据，得到节点特征X 和标签Y，邻接矩阵adjacency，训练集train_mask、验证集val_mask 以及测试集test_mask
        下面说的维度是以cora数据为例
        """
        # 尝试读取缓存数据
        if os.path.exists(self.save_file) and not self.rebuild:
            self.logger_info("Using Cached file: {}".format(self.save_file))
            _data = pickle.load(open(self.save_file, "rb"))
            return _data
        
        self.logger_info("Process data ...")

        # 读取数据
        items = [self.read_data(os.path.join(self.data_root, name)) for name in self.filenames]
        trainx, testx, semi_supervised_x, trainy, testy, semi_supervised_y, graph_adj, test_index = items

        sorted_test_index = sorted(test_index)  # test_index 是乱序, 排序
        if  'citeseer' in self.data_root:
            # Citeseer 图中有一些孤立的节点，导致没有连续的测试索引。我们需要识别它们，并将它们作为0向量添加到 `tx` 和 `ty`。
            # 经过检查, [2407, 2489, 2553, 2682, 2781, 2953, 3042, 3063, 3212, 3214, 3250, 3292, 3305, 3306, 3309] 不在test_index中
            len_test_indices = int(test_index.max() - test_index.min()) + 1
            tx_ext = np.zeros((len_test_indices, testx.shape[1]), dtype=np.float32)
            tx_ext[sorted_test_index - test_index.min(), :] = testx
            ty_ext = np.zeros((len_test_indices, testy.shape[1]), dtype=np.int32)
            ty_ext[sorted_test_index - test_index.min(), :] = testy
            testx, testy = tx_ext, ty_ext
        # 拼接完整数据
        x = np.concatenate((semi_supervised_x, testx), axis=0)                  # cora 1708 + 1000 =2708 特征向量
        y = np.concatenate((semi_supervised_y, testy), axis=0).argmax(axis=1)   # 1708 + 1000 =2708 标签向量 由one-hot编码转成0-6的标签
        
        # 归一化特征数据，使得每一行和为1
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        x = x / row_sums
        
        # 重新映射测试集. 样例,  test_index=[2692, ], sorted_test_index=[1708], 样例含义是将 1708 的特征向量放到 2692 的位置上
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]

        if self.use_semi_supervised:
            train_ids = np.arange(trainy.shape[0])  # cora=[0,...139] 140个元素
            val_ids = np.arange(trainy.shape[0], trainy.shape[0] + 500)  # cora=[140 - 640] 500个元素
            test_ids = test_index
        else:
            # 按 0.6 0.2 0.2 切分数据集
            n = y.shape[0]
            idx = [i for i in range(n)]
            #random.shuffle(idx)
            r0 = int(n * 0.6)
            r1 = int(n * 0.6)
            r2 = int(n * 0.8)

            train_ids = np.array(idx[:r0])
            val_ids = np.array(idx[r1:r2])
            test_ids = np.array(idx[r2:])

        #构建邻接矩阵
        adjacency = self.build_adjacency(graph_adj)
        _data = dict(x=x, y=y, adjacency=adjacency,
                    train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        # 将[x, y, adjacency, train_mask, val_mask, test_mask]缓存
        with open(self.save_file, "wb") as f:
            pickle.dump(_data, f)
        self.logger_info("Cached file: {}".format(self.save_file))
        return _data

    def print_data_info(self, data):
        # 打印图数据形状
        self.logger_info("Node's feature shape: ", data['x'].shape)
        self.logger_info("Node's label shape:   ", data['y'].shape)
        self.logger_info("Adjacency's shape:    ", data['adjacency'].shape)
        self.logger_info('train_ids num:        ', len(data['train_ids']))
        self.logger_info('val_ids num:          ', len(data['val_ids']))
        self.logger_info('test_ids num:         ', len(data['test_ids']))

    @staticmethod
    def build_adjacency(adj_dict):
        """
        根据邻接表创建 二维边矩阵

        原始邻接列表adj_dict 格式为 {index：[index_of_neighbor_nodes]}

        Return: 
            adj(二维边矩阵) : np.ndarray, shape=(2, edges)
        """
        num_nodes = len(adj_dict)
        src_nodes = []
        dst_nodes = []
        for src, dst in adj_dict.items():
            # 将邻接表中的节点，分别存入src_nodes和dst_nodes
            src_nodes.extend([src] * len(dst))
            dst_nodes.extend(dst)
        
        adj = np.array([src_nodes, dst_nodes], dtype=np.int64)
        
        return adj

    @staticmethod
    def read_data(path):
        """读取原始数据"""
        name = os.path.basename(path)
        if "test.index" in name:
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    def get_project_path(self):
        """获取项目根目录"""
        # xx/src/gcn
        path = os.path.dirname(os.path.abspath(__file__))
        # xx/src
        path = os.path.dirname(path)
        # xx
        path = os.path.dirname(path)
        return path

    def to_tensor(self, x, device='cpu'):
        """numpy --> torch.tensor"""
        return torch.from_numpy(x).to(device)


class GraphDataSet(Dataset):
    def __init__(self, data, target) -> None:
        self.data = data  # tensor
        self.target = target  # tensor

    def __getitem__(self, index):
        """
        返回的data是tensor
        因为模型只支持 list or np.ndarray
        因此在使用时需要将torch.tensor --> np.ndarray
        """
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)


if __name__=="__main__":
    # ['cora', 'citeseer', 'pubmed']
    # 这样可以单独测试Process_data函数
    config = {'data_path':'cora', "use_semi_supervised":False, 'device':'cpu', 'batch_size':4}
    graphdata = GraphData(config=config, rebuild=True)

    x, y, train_ids, val_ids, test_ids, adj = graphdata.get_data()
    input_dim, output_dim = graphdata.input_dim, graphdata.class_num

    # 创建数据加载器
    train_loader = DataLoader(GraphDataSet(train_ids, y[train_ids]), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(GraphDataSet(val_ids, y[val_ids]), batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(GraphDataSet(test_ids, y[test_ids]), batch_size=config['batch_size'], shuffle=False)

    n = 0
    for i, (data, target) in enumerate(train_loader):
        print(data, target)
        print(y[data])
        n += 1
        if n == 2:
            break