#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/11 09:30:08
@Desc :训练测试GCN模型
'''
import os
import sys
import time
import argparse
from loguru import logger
import torch

# 添加运行环境路径
sys.path.append('src/gcn')

# 导入训练函数
from src.gcn.train import trainer
from src.utils import get_log_path, format_seconds


# 训练参数设置，可以手动改变 default 值
parser = argparse.ArgumentParser(
    description='GCN训练脚本',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--data_path', type=str, default="cora", help='数据集路径, 如果是cora、citeseer、pubmed,填写名称即可.')
parser.add_argument('--device', type=str, default='cuda', help='默认使用GPU进行训练, cuda or cpu or cuda:0 ...')
parser.add_argument('--seed', type=int, default=72, help='随机种子.')
parser.add_argument('--epochs', type=int, default=200, help='训练轮数.')
parser.add_argument('--lr', type=float, default=0.1, help='学习率.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减.')
# 模型参数
parser.add_argument('--layers', type=int, default=2, help='GCN 卷积层数.')
parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层嵌入维度.')

parser.add_argument('--patience', type=int, default=50, help='早停轮数.')
parser.add_argument('--test_model', type=str, default='best', help='测试使用的模型[best or last], best是验证集最佳模型, last为最后一个epoch模型.')
parser.add_argument('--rebuild_data', type=str, default="False", help='是否重新构建数据集.')
parser.add_argument('--use_semi_supervised', type=str, default="True", help='是否采用半监督数据划分, 只对cora、citeseer、pubmed数据集生效.')
parser.add_argument('--is_draw', type=str, default="True", help="是否画实验结果图.")
parser.add_argument('--save_path', type=str, help='模型保存路径.')
args = parser.parse_args()

# 检查参数是否合法
if 'cuda' in args.device:
    assert torch.cuda.is_available(), RuntimeError('GPU不可用, 请检查设备是否支持GPU。或者将`--device`设置为cpu')
assert args.use_semi_supervised in ["True", "False"], ValueError("`--use_semi_supervised` 的值范围是 [True, False]")
assert args.rebuild_data in ["True", "False"], ValueError("`--rebuild_data` 的值范围是 [True, False]")
assert args.is_draw in ["True" or "False"], ValueError("`--is_draw` 的值范围是 [True, False]")
assert args.test_model in ["best", "last"], ValueError("`--test_model` 的值范围是 [best, last]")

# 将参数转为字典
config = vars(args)
config['use_semi_supervised'] = config["use_semi_supervised"] == "True"
config['rebuild_data'] = config["rebuild_data"] == "True"
config['is_draw'] = config["is_draw"] == "True"

if __name__ == '__main__':
    # 创建日志地址
    log_path, log_file = get_log_path('gcn')
    
    # 日志配置
    logger.remove()
    # 添加控制台sink，并自定义格式
    logger.add(sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True)
    logger.add(log_file, level="INFO", format="{time} | {message}")

    # 配置变量添加日志路径
    if config['save_path'] is None:
        config['save_path'] = log_path
    assert os.path.exists(config['save_path']), f'模型保存路径{config["save_path"]}不存在, 请检查路径是否正确, 或手动创建该路径。'

    print(config)

    t1 = time.time()
    # 开始训练
    trainer(config=config, logger_info=logger.info)
    logger.info('\n训练耗时: {}\n'.format(format_seconds(time.time() - t1)))



















