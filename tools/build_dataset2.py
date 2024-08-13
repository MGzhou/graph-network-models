#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/24 18:16:47
@Desc :自定义数据集, 可以用于GraphSAGE模型, 就除了邻接矩阵保存格式不一样，其他均相同
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

# 2 邻居矩阵, 例如第一行0和第二行1 表示节点0和节点1有边
adj_edge = np.array([
        [0,0,0,0, 1,1, 2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6],
        [2,3,5,6, 2,5, 0,3, 1,2,6, 2,3,5, 0,2,3, 1,3,5]
    ], dtype=np.int32)

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

dataset = dict(x=x, y=y, adjacency=adj_edge, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

# 保存
with open("test2.pkl", "wb") as fb:
    pickle.dump(dataset, fb)






