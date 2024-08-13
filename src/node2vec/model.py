#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/26 11:27:05
@Desc : Node2Vec model
'''
import torch
from torch import nn
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import networkx as nx
import itertools
import random
import numpy as np


class Node2Vec(object):
    def __init__(self, graph, p, q, walk_length, num_walks, workers=1, verbose=0, random_state=72):
        """paper: <https://arxiv.org/abs/1607.00653>"""
        self.G = graph                      # 图

        self.p = p                          # 返回参数
        self.q = q                          # 出入参数
        self.walk_length = walk_length      # 游走深度
        self.num_walks = num_walks          # 游走次数
        self.workers = workers              # 并行数量
        self.verbose = verbose              # 是否显示详细信息。日志等级。
        self.w2v = None                     # gensim的word2vec模型
        self.random_state = random_state    # 随机种子
        # 初始化转移概率
        self.preprocess_transition_probs()  

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
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def node2vec_walk(self, walk_length, start_node):
        """
        执行单次随机游走。
        从指定起始节点开始，进行指定长度的随机游走，返回节点序列。
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                if len(walk) == 1:
                    walk.append(
                        current_nerghbors[self.alias_sample(alias_nodes[current_node][0], alias_nodes[current_node][1])]
                    )
                else:
                    previous_node = walk[-2]
                    edge = (previous_node, current_node)
                    next_node = current_nerghbors[self.alias_sample(alias_edges[edge][0],   alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    
    def preprocess_transition_probs(self):
        """
        初始化随机游走的转移概率
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.create_alias_table(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        
        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes

    def get_alias_edge(self, t, v):
        """
        2阶随机游走，顶点间的转移概率
        :param t: 上一顶点
        :param v: 当前顶点
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx，无权图权重设为1
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
                            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.create_alias_table(normalized_probs)

    def create_alias_table(self, probs):
        """
        构建 Alias 别名表
        :param probs: sum(probs)=1
        :return: accept, alias
        """
        L = len(probs)
        accept, alias = [0] * L,  [0] * L
        small, large = [], []
        for i, prob in enumerate(probs):
            accept[i] = prob * L
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = probs[small_idx]
            alias[small_idx] = large_idx
            probs[large_idx] = probs[large_idx] - (1 - probs[small_idx])
            if probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        return accept, alias

    def alias_sample(self, accept, alias):
        """
        alias 采样
        :return: sample index
        """
        N = len(accept)
        i = int(np.random.random()*N)
        r = np.random.random()
        if r < accept[i]:
            return i
        else:
            return alias[i]

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



class Node2VecClassifier(nn.Module):
    def __init__(self, 
                 embed_size, 
                 out_dim, 
                 p,
                 q,
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
        Node2Vec 分类模型

        Args:
            embed_size (int):   节点嵌入向量维度大小
            out_dim (int):      分类输出维度大小
            p (float): 返回参数，其决定了算法返回到前一个节点的概率。当 p > 1 时，算法倾向于回到先前访问过的节点；而当 p < 1 时，则更倾向于探索未曾或较少访问的邻居节点。
            q (float): 出入参数，其影响了算法是倾向于探索远离起始点的节点(q < 1), 还是靠近起始点的节点(q > 1)。
            walk_length (int):  随机游走长度(深度)
            num_walks (int):    每个节点随机游走次数, 即每个节点生成的句子个数。
            wv_window_size (int):  Word2Vec的窗口大小。
            wv_min_count (int):    词频少于min_count次数的单词会被丢弃掉，默认值为0。
            wv_epochs (int):    Word2Vec 训练轮次。
            workers (int):      Word2Vec 与 随机游走 生成并行数量. Defaults to 3.
            verbose (int):      是否显示详细信息。日志等级。. Defaults to 0.
            random_state (int): 随机种子. Defaults to 72.
        """
        super(Node2VecClassifier, self).__init__()

        self.embeddings = None
        self.node2vec = None

        # 参数
        self.p = p
        self.q = q
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
        self.node2vec = Node2Vec(
            graph=graph, 
            p=self.p,
            q=self.q,
            walk_length=self.walk_length, 
            num_walks=self.num_walks, 
            workers=self.workers, 
            verbose=self.verbose, 
            random_state=self.random_state
        )
        # 训练deepwalk模型
        self.node2vec.fit(
            embed_size=self.embed_size, 
            window=self.wv_window_size, 
            n_jobs=self.workers, 
            epochs=self.wv_epochs, 
            min_count=self.wv_min_count
        )
        # 获取节点嵌入向量
        self.embeddings = self.node2vec.get_embeddings()
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32).to(self.device)
        print("end...\n")



"""
控制随机游走的行为参数
返回参数 p (Return Parameter): 这个参数决定了算法返回到前一个节点的概率。当 p > 1 时，算法倾向于回到先前访问过的节点；而当 p < 1 时，则更倾向于探索未曾或较少访问的邻居节点。
出入参数 q (In-out Parameter): 这个参数影响了算法是倾向于探索远离起始点的节点(q < 1), 还是靠近起始点的节点(q > 1)。

当 p 和 q 均为 1 时, Node2Vec 变成了传统的无偏随机游走。等同于 DeepWalk。
当 p > 1 且 q > 1 时，算法更倾向于生成类似于 BFS 的游走序列。
当 p < 1 且 q < 1 时，算法则倾向于生成 DFS 类型的游走序列。
当 p < 1 且 q > 1 时，算法既会探索较远的节点，也会回到先前访问过的节点，从而产生较为复杂的行为。

论文并没有直接根据转移概率采样，而是使用了 Alias Sampling（别名采样）方法来高效地实现随机游走。
Alias Sampling（别名采样）是一种用于高效从离散概率分布中采样的方法。它主要用于加速从具有离散概率分布的随机变量中采样的过程，尤其在概率分布不均匀的情况下特别有效。
"""