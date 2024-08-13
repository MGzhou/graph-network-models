#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_set import GraphData, GraphDataSet
from model import GraphSAGE
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
        x, y, train_ids, val_ids, test_ids, adj = graphdata.get_data()
        input_dim, output_dim = graphdata.input_dim, graphdata.class_num
    else:
        input_dim, output_dim, x, y, train_ids, val_ids, test_ids, adj = mini_data


    # 创建数据加载器
    train_loader = DataLoader(GraphDataSet(train_ids, y[train_ids]), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(GraphDataSet(val_ids, y[val_ids]), batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(GraphDataSet(test_ids, y[test_ids]), batch_size=config['batch_size'], shuffle=False)

    #------------------------- 模型定义 ----------------------------#
    model = GraphSAGE(
        in_dim=input_dim, 
        hidden_dim=config['hidden_dim'], 
        out_dim=output_dim, 
        k=config['k'],
        num_sample=config['num_sample'],
        agg=config['agg'],
        sub_batch_size=config['sub_batch_size'],
        dropout=config['dropout'], 
        concat=config['concat'],
        activation=config['activation'], 
        bias=config['bias'],
        graph_type=config['graph_type'],
        device=config['device']
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
    best_model_path = os.path.join(config['save_path'], "best_model.pth")               # val_acc best模型
    last_epoch_model_path = os.path.join(config['save_path'], "last_epoch_model.pth")   # 最后一个epoch模型

    for epoch in range(config['epochs']):
        model.train()
        batch_loss = []
        batch_train_acc = []
        # 批次训练
        for i, (data, label) in enumerate(train_loader):
            data, label = data.numpy(), label.to(config['device'])
            output = model(data, adj, x)
            optimizer.zero_grad()  # 清零梯度
            output = F.log_softmax(output, dim=1)
            loss = criterion(output, label)
            
            loss.backward()        # 反向传播，计算梯度
            optimizer.step()       # 更新参数
            batch_loss.append(loss.item())
            # 评估
            predict_y = output.max(1)[1]  # 返回每一行的最大值中索引（返回最大元素在各行的列索引）
            train_acc = torch.eq(predict_y, label).float().mean()
            batch_train_acc.append(train_acc.item())
        loss = np.average(batch_loss)
        train_acc = np.average(batch_train_acc)

        # 评估
        val_acc = evaluate(config, model, adj, x, val_loader)  # 计算当前模型在验证集上的准确率

        # 记录 loss 和 val_acc
        loss_history.append(loss)
        val_acc_history.append(val_acc)

        # 检查是否执行早停
        if val_acc - delta > best_val_acc:
            logger_info("Epoch {:03d}/{}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}, best epoch".format(
                epoch, config['epochs'], loss, train_acc, val_acc))
            best_val_acc = val_acc
            bad_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)
        else:
            logger_info("Epoch {:03d}/{}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
                epoch, config['epochs'], loss, train_acc, val_acc))

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

    test_acc, output_emb, test_y = evaluate(config, model, adj, x, test_loader, is_test=True)
    logger_info(f"Test accuarcy: {test_acc.item():.4f}", )

    # 画loss和acc的曲线图，散点图，默认画
    if config['is_draw']:
        plot_loss_with_acc(loss_history, val_acc_history, config['save_path'])
        test_x = [i for i in range(len(test_y))]
        plot_embeddings(test_x, test_y, output_emb, config['save_path'])


def evaluate(config, model, adj, x, data_loader, is_test=False):
    """评估函数

    Args:
        model: 模型
        adj (torch.sparse.tensor): 边矩阵
        x (torch.tensor): 所有节点特征
        data_loader (data.DataLoader): 数据加载器
        is_test (bool, optional): 是否是测试集. Defaults to False.

    Returns:
        acc: 准确率
        output: 预测最后输出向量
        y: 评估的节点的真实标签
    """
    model.eval()  
    with torch.no_grad():
        batch_output = []
        batch_label = []
        for i, (data, label) in enumerate(data_loader):
            data, label = data.numpy(), label.to(config['device'])
            output = model(data, adj, x)
            # 保存每个批次的结果，最后一起统计准确率
            batch_output.append(output.cpu())
            batch_label.append(label.cpu())

    output = torch.cat(batch_output, dim=0)
    y = torch.cat(batch_label, dim=0)
    # 计算准确率
    predict_y = output.max(1)[1]
    acc = torch.eq(predict_y, y).float().mean()

    if not is_test:
        return acc
    else:
        return acc, output.numpy(), y.numpy()

if __name__ == '__main__':
    config = {
        'data_path': 'cora', 
        'use_semi_supervised':False,
        'device': 'cuda', 
        'seed': 72, 
        'epochs': 2, 
        'batch_size':2,
        'lr': 0.005, 
        'weight_decay': 0.0005, 
        'hidden_dim': 4, 
        'k': 2, 
        'num_sample': 3, 
        'agg': 'mean', 
        'sub_batch_size': 2,
        'dropout': 0.0, 
        'concat': False, 
        'activation': True, 
        'bias': False, 
        'graph_type': 'edge', 
        'patience': 50, 
        'test_model': 'last', 
        'rebuild_data':False,
        'is_draw': True, 
        'save_path': '../../logs/graphsage-test'
    }
    adj_edge = np.array([
        [0,0,0,0,1,1,1,1, 2,2,2, 3,3, 4,4, 5,5],
        [1,2,3,5,0,2,3,4, 0,1,3, 0,2, 1,5, 0,4]
    ])
    #　输入特征不直接转移到GPU上，而是在模型中转移
    x = torch.tensor([[-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091],
        [-0.7121,  0.3037, -0.7773, -0.2515, -0.2223,  1.6871,  0.2284,  0.4676],
        [-0.6970, -1.1608,  0.6995,  0.1991,  0.8657,  0.2444, -0.6629,  0.8073],
        [ 1.1017, -0.1759, -2.2456, -1.4465,  0.0612, -0.6177, -0.7981, -0.1316],
        [ 1.8793, -0.0721,  0.1578, -0.7735,  0.1991,  0.0457,  0.1530, -0.4757],
        [-0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373,  1.5748, -0.6298]])  # .to(config['device'])
    
    # input_dim, output_dim, x, y, train_ids, val_ids, test_ids, adj

    y = torch.tensor([0, 0, 1, 1, 0, 1])
    train_ids = torch.tensor([0, 1, 2])
    val_ids = torch.tensor([3])
    test_ids = torch.tensor([4, 5])
    input_dim, output_dim = 8, 2
    mini_data = [input_dim, output_dim, x, y, train_ids, val_ids, test_ids, adj_edge]

    trainer(config=config, mini_data=mini_data)











