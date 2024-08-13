#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:06:40
@Desc :读取和处理数据集的类文件
'''


import os
import pickle
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class GraphData(object):
    def __init__(self, config, rebuild=False, logger_info=print):
        """数据处理, 加载.

        处理之后的数据包括如下几部分(cora为例)：
            * x: 节点的特征 (2708,1433)
            * y: 节点的标签, 总共包括7个类别 (2078,)
            * adjacency:    邻接矩阵       (2708, 2708)
            * train_mask:   训练集掩码索引 (2708,)
            * val_mask:     验证集掩码索引 (2708,)
            * test_mask:    测试集掩码索引 (2708,)

        Args:
            config (dict): 配置字典
            rebuild (boolean): 是否需要重构数据集, 默认读取存在的缓存。当设为True时, 就算存在缓存数据也会重建数据
            logger_info (function): 日志打印函数
        """
        self.rebuild = rebuild
        self.device = config['device']
        self.use_semi_supervised = config['use_semi_supervised']
        self.data_root = config['data_path']
        self.logger_info = logger_info

        # 已有公开数据集
        if self.data_root in ['cora', 'citeseer', 'pubmed']:
            # 文件名称
            self.filenames = ["ind.{}.{}".format(self.data_root, name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
            # 名称转路径
            self.data_root = os.path.join(self.get_project_path(), 'data', self.data_root)
            # 缓存数据路径
            if self.use_semi_supervised:
                self.save_file = os.path.join(self.data_root, "gat_processed_data_semi.pkl")
            else:
                self.save_file = os.path.join(self.data_root, "gat_processed_data_sup.pkl")
        elif not os.path.exists(self.data_root):
            raise ValueError("data_root {} does not exists.".format(self.data_root))
        else:
            # 自定义数据集文件, 需要绝对路径, 如 xx/xxx.pkl
            is_file = os.path.isfile(self.data_root)
            if is_file:
                self.rebuild = False
                self.save_file = self.data_root
            else:
                raise ValueError("data_root {} does not a file.".format(self.data_root))

    def get_data(self):
        """
        获取输入模型的数据, 包括节点特征、标签、训练集、验证集、测试集, 邻接矩阵

        Return:
            数据为torch.Tensor, 并会转移到指定的(CPU或GPU)设备上
            其中 tensor_adjacency type: torch.float32
        """
        dataset = self.pre_process()
        self.print_data_info(dataset)  # 打印数据集信息

        tensor_x = self.to_tensor(dataset['x']) # 节点特征，cora=(2708,1433)
        tensor_y = self.to_tensor(dataset['y']) # 节点标签，cora=(2708,)
        _, input_dim = tensor_x.shape   # 节点特征维度, cora=(2708,1433)
        class_num = tensor_y.max().item() + 1   # 类别数量  cora=7

        # 索引
        tensor_train_mask = self.to_tensor(dataset['train_ids'])   # cora数据集前140个为True
        tensor_val_mask = self.to_tensor(dataset['val_ids'])       # cora数据集 140 - 639  500个为True
        tensor_test_mask = self.to_tensor(dataset['test_ids'])     # cora数据集 1708 - 2707 1000个

        adjacency = dataset['adjacency']                            # 邻接矩阵, cora=(4732,4732)
        adjacency = torch.FloatTensor(adjacency).to(self.device)

        self.input_dim, self.class_num = input_dim, class_num
        
        return tensor_x, tensor_y, tensor_train_mask, tensor_val_mask, tensor_test_mask, adjacency

    def pre_process(self):
        """
        处理数据，得到节点特征X 和标签Y，邻接矩阵adjacency，训练集train_mask、验证集val_mask 以及测试集test_mask
        下面说的维度是以cora数据为例
        缓存数据格式
            x np.ndarray, 
            y np.ndarray,
            train_mask np.ndarray, 
            val_mask np.ndarray, 
            test_mask np.ndarray, 
            adjacency np.ndarray
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
        # 训练集，测试集，测试集索引
        train_index = np.arange(trainy.shape[0])  # [0,...139] 140个元素
        val_index = np.arange(trainy.shape[0], trainy.shape[0] + 500) # [140 - 640] 500个元素
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
        x = np.concatenate((semi_supervised_x, testx), axis=0)  # 1708 + 1000 =2708 特征向量
        y = np.concatenate((semi_supervised_y, testy), axis=0).argmax(axis=1) # 1708 + 1000 =2708 标签向量 由one-hot编码转成0-6的标签
        
        # 归一化特征数据，使得每一行和为1
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        x = x / row_sums
        # 重新映射测试集. 样例,  test_index=[2692, ], sorted_test_index=[1708], 样例含义是将 1708 的特征向量放到 2692 的位置上
        x[test_index] = x[sorted_test_index]  # 按照给定test的下标, 将特征向量重新排列, 不排序也没有关系，因为GAT是所有节点一起评估的，不涉及批次，因此求平均时无影响。
        y[test_index] = y[sorted_test_index]

        if self.use_semi_supervised:
            train_ids = train_index # cora=[0,...139] 140个元素
            val_ids = val_index     # cora=[140 - 640] 500个元素
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


        # # 训练集，测试集，测试集掩码向量
        # num_nodes = x.shape[0]
        # train_mask = np.zeros(num_nodes, dtype=np.bool_)  # 生成零向量
        # val_mask = np.zeros(num_nodes, dtype=np.bool_)
        # test_mask = np.zeros(num_nodes, dtype=np.bool_)
        # train_mask[train_index] = True  # cora前140个元素为训练集
        # val_mask[val_index] = True      # 140 -639 500个
        # test_mask[test_index] = True    # 1708-2708 1000个元素
        
        #构建邻接矩阵
        adjacency = self.build_adjacency(graph_adj)
        
        self.logger_info('test_index shape:     ',test_index.shape)
        self.logger_info('train_index shape:    ',train_index.shape)
        self.logger_info('val_index shape:      ',val_index.shape)  

        _data = dict(x=x, y=y, adjacency=adjacency,
                    train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        # 将[x, y, adjacency, train_mask, val_mask, test_mask]缓存
        with open(self.save_file, "wb") as f:
            pickle.dump(_data, f)
        self.logger_info("Cached file: {}".format(self.save_file))

        return _data

    def print_data_info(self, data):
        # 打印图数据形状
        self.logger_info("Node's feature shape:         ", data['x'].shape)
        self.logger_info("Node's label shape:           ", data['y'].shape)
        self.logger_info("Adjacency's shape:            ", data['adjacency'].shape)
        self.logger_info("Number of training nodes:     ", len(data['train_ids']))
        self.logger_info("Number of validation nodes:   ", len(data['val_ids']))
        self.logger_info("Number of test nodes:         ", len(data['test_ids']))

    @staticmethod
    def build_adjacency(adj_dict):
        """
        根据邻接表创建邻接矩阵

        原始邻接列表adj_dict 格式为 {index：[index_of_neighbor_nodes]}

        Return: 
            adj(邻接矩阵) : np.int32, shape = (num_nodes, num_nodes)
        """
        num_nodes = len(adj_dict)
        # 创建一个全零矩阵，cora =（2078，2078）
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)

        # 创建 无向 邻接矩阵
        for src, dst in adj_dict.items():
            for v in dst:
                adj[src][v] = 1  
                adj[v][src] = 1
        
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

    def to_tensor(self, x):
        """numpy --> torch.tensor"""
        return torch.from_numpy(x).to(self.device)

if __name__=="__main__":
    # ['cora', 'citeseer', 'pubmed']
    # 这样可以单独测试Process_data函数
    config = {'data_path':'pubmed',"use_semi_supervised":True, 'device':'cpu'}
    a = GraphData(config=config, rebuild=True)
    x, y, train_mask, val_mask,test_mask, adjacency = a.get_data()

    print(a.class_num)
    print(a.input_dim)
