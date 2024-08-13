#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:06:23
@Desc :工具函数文件
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_loss_with_acc(loss_history, val_acc_history, save_path=None):
    """画loss与准确率的曲线图

    Args:
        loss_history (list,np): 损失历史
        val_acc_history (list, np): 验证集acc历史
        save_path (str, optional): 图像保存路径. Defaults to None.
    """
    fig = plt.figure()
    
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(range(len(loss_history)), loss_history,
                      c=np.array([255, 71, 90]) / 255., label='train_loss')
    ax1.set_ylabel('Loss')
    
    # 坐标系ax2画曲线2
    ax2 = ax1.twinx()
    line2, = ax2.plot(range(len(val_acc_history)), val_acc_history,
                      c=np.array([79, 179, 255]) / 255., label='val_acc')
    ax2.set_ylabel('ValAcc')
    
    ax1.set_xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')  # 或者选择其他位置如'best' 'upper left' 'lower left'

    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_acc.png'))
    else:
        plt.show()

def plot_embeddings(X, Y, embeddings, save_path=None):
    """
    画散点图

    Args:
        X (list): 节点索引列表
        Y (list): 节点类别列表
        embeddings (np or torch): 所有节点嵌入向量矩阵

    Sample:
        >>> X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> Y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        >>> embeddings = torch.randn(10, 128)
        >>> plot_embeddings(X, Y, embeddings)
    """
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    plt.figure(figsize=(15, 10))
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'embedding.png'))
    else:
        plt.show()

if __name__=="__main__":
    import torch
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    embeddings = torch.randn(10, 16)
    plot_embeddings(X, Y, embeddings)