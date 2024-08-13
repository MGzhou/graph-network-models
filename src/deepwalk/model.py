#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/26 11:27:05
@Desc : DeepWalk model
'''
import torch
from torch import nn
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import networkx as nx
import itertools
import random
import numpy as np

class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, workers=1, verbose=0, random_state=72):
        self.G = graph                      # 图
        self.walk_length = walk_length      # 游走深度
        self.num_walks = num_walks          # 游走次数
        self.workers = workers              # 并行数量
        self.verbose = verbose              # 是否显示详细信息。日志等级。
        self.w2v = None                     # gensim的word2vec模型
        # self.embeddings = None              # 嵌入向量
        self.random_state = random_state    # 随机种子
        # 存储由图G生成的w2v训练数据
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.workers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        """
        训练Word2Vec模型。
        使用生成的训练数据集训练Word2Vec模型，得到节点的嵌入向量。
        """
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)  # 词频少于min_count次数的单词会被丢弃掉，默认值为0。
        kwargs["vector_size"] = embed_size     # 嵌入向量的维度。
        kwargs["sg"] = 1                # skip gram
        kwargs["hs"] = 1                # deepwalk use Hierarchical Softmax
        kwargs["workers"] = n_jobs      # 并行数量。
        kwargs["window"] = window       # Word2Vec的窗口大小。
        kwargs["epochs"] = epochs         # 训练迭代次数。
        kwargs["seed"] = self.random_state

        self.w2v = Word2Vec(**kwargs)
        
    def get_train_data(self, walk_length, num_walks, workers=1, verbose=0):
        """
        生成训练数据集。
        通过并行的方式对每个节点进行随机游走，生成用于训练的序列数据。
        """
        if num_walks % workers == 0:
            num_walks = [num_walks//workers]*workers
        else:
            num_walks = [num_walks//workers]*workers + [num_walks % workers]

        nodes = list(self.G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self.simulate_walks)(nodes, num, walk_length) for num in num_walks
        )
        dataset = list(itertools.chain(*results))
        return dataset

    def simulate_walks(self, nodes, num_walks, walk_length,):
        """
        模拟随机游走。
        对每个节点进行指定次数的随机游走，生成游走序列。
        """
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                    walks.append(self.deep_walk(walk_length=walk_length, start_node=v))
        return walks

    def deep_walk(self, walk_length, start_node):
        """
        执行单次随机游走。
        从指定起始节点开始，进行指定长度的随机游走，返回节点序列。
        """
        G = self.G

        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                walk.append(random.choice(current_nerghbors))
            else:
                break
        return walk

    def get_embeddings(self):
        """
        获取节点嵌入向量。
        """
        if self.w2v:
            embeddings = []
            # 节点已经按照顺序排列，因此可以按照顺序获取嵌入向量
            for node in self.G.nodes():
                embeddings.append(self.w2v.wv[node])
            embeddings = np.array(embeddings)
            return embeddings
        else:
            print("Please train the model first")
            return None
    
    def get_node_embeddings(self, node_id):
        """
        获取指定节点的嵌入向量。
        """
        return self.w2v.wv[node_id]



class DeepWalkClassifier(nn.Module):
    def __init__(self, 
                 embed_size, 
                 out_dim, 
                 walk_length, 
                 num_walks, 
                 wv_window_size=5, 
                 wv_min_count=0,
                 wv_epochs=5,
                 workers=3, 
                 verbose=0, 
                 random_state=72,
                 device="cpu"
    ) -> None:
        """
        DeepWalk 分类模型

        Args:
            embed_size (int):   节点嵌入向量维度大小
            out_dim (int):      分类输出维度大小
            walk_length (int):  随机游走长度(深度)
            num_walks (int):    每个节点随机游走次数, 即每个节点生成的句子个数。
            wv_window_size (int):  Word2Vec的窗口大小。
            wv_min_count (int):    词频少于min_count次数的单词会被丢弃掉，默认值为0。
            wv_epochs (int):    Word2Vec 训练轮次。
            workers (int):      Word2Vec 与 随机游走 生成并行数量. Defaults to 3.
            verbose (int):      是否显示详细信息。日志等级。. Defaults to 0.
            random_state (int): 随机种子. Defaults to 72.
        """
        super(DeepWalkClassifier, self).__init__()

        self.embeddings = None
        self.deepwalk = None

        # 参数
        self.embed_size = embed_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.wv_window_size = wv_window_size
        self.wv_min_count = wv_min_count
        self.wv_epochs = wv_epochs
        self.workers = workers
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        
        # 一层全连接层分类
        self.classer = nn.Linear(embed_size, out_dim)

    def forward(self, adj):
        """
        前向传播函数

        Args:
            adj (np.ndarray): 邻接矩阵
        """
        if self.embeddings is None:
            self.train_embeddings(adj)
        
        x = self.classer(self.embeddings)

        return x

    def train_embeddings(self, adj):
        """
        训练deepwalk模型并获取节点嵌入向量
        """
        print("Training node embedding model...")
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph())
        self.deepwalk = DeepWalk(
            graph=graph, 
            walk_length=self.walk_length, 
            num_walks=self.num_walks, 
            workers=self.workers, 
            verbose=self.verbose, 
            random_state=self.random_state
        )
        # 训练deepwalk模型
        self.deepwalk.fit(
            embed_size=self.embed_size, 
            window=self.wv_window_size, 
            n_jobs=self.workers, 
            epochs=self.wv_epochs, 
            min_count=self.wv_min_count
        )
        # 获取节点嵌入向量
        self.embeddings = self.deepwalk.get_embeddings()
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32).to(self.device)
        print("end...\n")