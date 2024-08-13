#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from data_set import GraphData
from model import GAT
from utils import plot_loss_with_acc, plot_embeddings

def trainer(config, mini_data=None, logger_info=print):
    """
    训练函数

    Args:
        config: 参数字典
        mini_data: 用于直接传入处理好的数据
        logger: 训练日志打印函数
    """
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if 'cuda' in config['device']:
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 加载数据
    if mini_data is None:
        graphdata = GraphData(config=config, rebuild=config['rebuild_data'])
        # 所有节点特征、所有标签、训练集节点索引、验证集节点索引、测试集节点索引, 邻接矩阵
        x, y, train_mask, val_mask, test_mask, adj = graphdata.get_data()
        input_dim, output_dim = graphdata.input_dim, graphdata.class_num
    else:
        input_dim, output_dim, x, y, train_mask, val_mask, test_mask, adj = mini_data

    #------------------------- 模型定义 ----------------------------#
    model = GAT(
        in_dim=input_dim, 
        hidden_dim=config['hidden_dim'],
        out_dim=output_dim,
        num_heads=config['num_heads'],
        dropout=config['dropout'], 
        alpha=config['alpha'], 
        residual=config['residual'], 
        bias=config['bias']
    ).to(config['device'])
    # 优化器
    criterion = torch.nn.NLLLoss().to(config['device'])  # loss
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    #---------------------------- 训练 ---------------------------------
    loss_history = []       # 记录loss
    val_acc_history = []    # 记录验证集准确率
    # 早停
    patience = config['patience']  # 早停次数
    delta = 0           # 只有当验证集上的acc变化超过delta时，才认为有改善
    best_val_acc = 0    # 最佳验证集acc值
    bad_counter = 0     # 计数器，用于记录验证集acc没有改善的次数
    early_stop = False
    
    # 保存模型地址
    best_model_path = os.path.join(config['save_path'], "best_model.pth")                   # val_acc best模型
    last_epoch_model_path = os.path.join(config['save_path'], "last_epoch_model.pth")       # 最后一个epoch模型
    
    train_y = y[train_mask]  # 训练集标签

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()   # 清零梯度
        output = model(x, adj)  # 前向传播
        # output = F.log_softmax(output, dim=1)  # 对输出进行log_softmax
        train_mask_output = output[train_mask]   # 只选择训练节点计算loss
        loss = criterion(train_mask_output, train_y)    # 计算损失值  

        loss.backward()     # 反向传播，计算梯度
        optimizer.step()    # 更新参数

        # 评估
        train_acc = evaluate(model, adj, x, y, train_mask)  # 计算当前模型训练集上的准确率
        val_acc = evaluate(model, adj, x, y, val_mask)      # 计算当前模型在验证集上的准确率

        # 记录 loss 和 val_acc
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())

        # 检查是否执行早停
        if val_acc - delta > best_val_acc:
            logger_info("Epoch {:03d}/{}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}, best epoch".format(
            epoch, config['epochs'], loss.item(), train_acc.item(), val_acc.item()))
            best_val_acc = val_acc
            bad_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)
        else:
            logger_info("Epoch {:03d}/{}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, config['epochs'], loss.item(), train_acc.item(), val_acc.item()))

            bad_counter += 1
            # 执行早停判断
            if bad_counter >= patience:
                early_stop = True
                break
        
    # 保存最后一个epoch模型
    torch.save(model.state_dict(), last_epoch_model_path)
    if early_stop:
        logger_info(f"训练{patience}个epoch没有改善, 实行早停.")
    else:
        logger_info("训练完成.")

    # 保存参数config字典
    with open(os.path.join(config['save_path'], 'config.json'), 'w', encoding='utf-8') as fw:
        json.dump(config, fw, ensure_ascii=False, indent=4)
    
    #------------------------- 测试 ----------------------------#
    # 加载最佳模型
    if config["test_model"] == "best":
        model.load_state_dict(torch.load(best_model_path))

    test_acc, output_emb, test_y = evaluate(model, adj, x, y, test_mask, is_test=True)
    logger_info(f"Test accuarcy: {test_acc.item():.4f}", )

    # 画loss和acc的曲线图，散点图，默认画
    if config['is_draw']:
        plot_loss_with_acc(loss_history, val_acc_history, config['save_path'])
        test_x = [i for i in range(len(test_y))]
        plot_embeddings(test_x, test_y, output_emb, config['save_path'])


def evaluate(model, adjacency, x, y, mask, is_test=False):
    """评估函数

    Args:
        model: 模型
        adjacency (torch.sparse.tensor): 邻居矩阵
        x (torch.tensor): 所有节点特征
        y (torch.tensor): 所有节点标签
        mask (torch.tensor): 数据集节点索引列表
        is_test (bool, optional): 是否是测试集. Defaults to False.

    Returns:
        acc: 准确率
        mask_output: 预测最后输出向量
        y: 评估的节点的真实标签
    """
    # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰
    model.eval()  
    with torch.no_grad():
        output = model(x, adjacency)
        mask_output = output[mask]
        acc = accuracy(mask_output, y[mask])
    if not is_test:
        return acc
    else:
        return acc, mask_output.cpu().numpy(), y[mask].cpu().numpy()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == '__main__':
    config = {
        'device': 'cuda', 
        'seed': 72, 
        'epochs': 2000, 
        'lr': 0.005, 
        'weight_decay': 0.0005, 
        'num_heads': 8, 
        'hidden_dim': 8, 
        'dropout': 0.6, 
        'alpha': 0.2, 
        'residual': False, 
        'activation': True, 
        'bias': False, 
        'patience': 50, 
        'test_model': 'best', 
        'rebuild_data':False,
        'is_draw': False, 
        'data_path': 'cora', 
        'save_path': '../../logs/gat-test'
    }
    in_dim = 8
    out_dim = 2
    
    x = torch.tensor([[-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091],
        [-0.7121,  0.3037, -0.7773, -0.2515, -0.2223,  1.6871,  0.2284,  0.4676],
        [-0.6970, -1.1608,  0.6995,  0.1991,  0.8657,  0.2444, -0.6629,  0.8073],
        [ 1.1017, -0.1759, -2.2456, -1.4465,  0.0612, -0.6177, -0.7981, -0.1316],
        [ 1.8793, -0.0721,  0.1578, -0.7735,  0.1991,  0.0457,  0.1530, -0.4757],
        [-0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373,  1.5748, -0.6298]]).to(config['device'])
    y = y = torch.tensor([0, 0, 1, 1, 0, 1]).to(config['device'])
    train_mask = torch.tensor([True, True, True, False, False,False]).to(config['device'])
    val_mask = torch.tensor([False, False, False, True, False, False]).to(config['device'])
    test_mask = torch.tensor([False, False, False, False, True, True]).to(config['device'])
    adj = [
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1]
    ]
    # 可以转为稀疏矩阵，不转为也没有出错
    adjacency = torch.FloatTensor(adj).to(config['device'])

    import pickle
    with open("/data/zmp/progect/LLM/gnn-models-alpha/references/pyGAT-master/data/cora-gat-mini-data.pkl", 'rb') as f:
        mini_data = pickle.load(f)
    # mini_data = [in_dim, out_dim, x, y, train_mask, val_mask, test_mask, adjacency]
    for i in range(len(mini_data)):
        if i >=2:
            mini_data[i] = mini_data[i].to(config['device'])

    # 检查保存路径
    trainer(config=config, mini_data=mini_data, logger_info=print)















