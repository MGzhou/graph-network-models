#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/24 18:16:47
@Desc :自定义数据集, 可以用于DeepWalk、Node2Vec、GCN、GAT模型
        如果图太大, 有些模型所需显存多大，可能运行不了。
'''

import pickle
import numpy as np

# 例子

n = 5  # 节点数量

class_num = 2  # 类别数量

# 1 节点特征, np.float32
x = [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
x = np.array(x, dtype=np.float32)

# 2 邻居矩阵, 1 表示节点有边
adj = [
        [0, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
    ]
adj = np.array(adj, dtype=np.float32)

# 3 标签
y = [0, 1, 1, 0, 1, 0, 1]    # 如果标签不是0开头的id格式，如果不是需要转换
y = np.array(y, dtype=np.int64)

# 4 训练集索引
train_ids = [0,1]
train_ids = np.array(train_ids, dtype=np.int64)

# 5 验证集索引，如果数量少，可以设置为和train一样
val_ids = [2, 3]
val_ids = np.array(val_ids, dtype=np.int64)

# 6 测试集索引
test_ids = [5,6]
test_ids = np.array(test_ids, dtype=np.int64)

dataset = dict(x=x, y=y, adjacency=adj, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

# 保存
with open("test.pkl", "wb") as fb:
    pickle.dump(dataset, fb)






