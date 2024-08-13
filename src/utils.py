#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/10 17:44:21
@Desc :None
'''
import os
import sys
from datetime import datetime

def format_seconds(seconds):
    """
    秒数格式化函数，将秒数转换为年、月、日、小时、分钟和秒的格式.
    
    一年按365天计算，一个月按30天计算.

    Args:
        seconds (float): 需要格式化的秒数

    Returns:
        str: 格式化后的时间字符串
    
    Sample:
        >>> format_seconds(3661)
        '1小时 1分钟 1秒'
        >>> format_seconds(61)
        '1分钟 1秒'
        >>> format_seconds(1)
        '1秒'
    """
   # 定义年、月、日的基本秒数
    seconds_per_year = 365 * 24 * 60 * 60  # 一年的秒数
    seconds_per_month = 30 * 24 * 60 * 60  # 一个月的秒数

    # 分解秒数到年、月、日、小时、分钟和秒
    years, remaining = divmod(seconds, seconds_per_year)
    months, remaining = divmod(remaining, seconds_per_month)
    days, remaining = divmod(remaining, 24 * 60 * 60)
    hours, remaining = divmod(remaining, 60 * 60)
    minutes, seconds = divmod(remaining, 60)

    # 构建结果字符串
    parts = []
    if years > 0:
        parts.append(f"{int(years)}年")
    if months > 0 or years > 0:
        parts.append(f"{int(months)}月")
    if days > 0 or months > 0 or years > 0:
        parts.append(f"{int(days)}日")
    if hours > 0 or days > 0 or months > 0 or years > 0:
        parts.append(f"{int(hours)}小时")
    if minutes > 0 or hours > 0 or days > 0 or months > 0 or years > 0:
        parts.append(f"{int(minutes)}分钟")
    parts.append(f"{int(seconds)}秒")

    return " ".join(parts)

def get_log_path(model_name="model"):
    """
    创建日志文件路径.

    Args:
        model_name (str): 模型名称，默认为"model"

    Returns:
        log_path (str): 日志文件夹路径
        log_file (str): 日志文件路径
    
    Sample:
        >>> get_log_path('gcn')
        'xxx/logs/gcn-2024-07-11_11-47-54_147'
        'xxx/logs/gcn-2024-07-11_11-47-54_147/log.log'
    """
    # 获取当前时间
    now = datetime.now()
    # 格式化日期和时间，包括毫秒
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S_") + f"{now.microsecond // 1000:03}"
    # 获取项目路径
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 创建日志文件路径
    log_file = os.path.join(root_path, f'logs/{model_name}-{current_time}', "log.log")
    log_path = os.path.dirname(log_file)
    return log_path, log_file


def add_path_to_sys(path=None, folder_level=0):
    """
    将项目目录添加到系统路径中，确保项目内的模块可以被导入。

    Args:
        path (str, optional): 路径. Defaults to None.
        folder_level (int, optional): 路径上调层级. Defaults to 0.
    
    Samples:
        >>> add_path_to_sys(path="D:/a/b", folder_level=1)
            添加路径 D:/a 
        >>> add_path_to_sys(path="D:/a/b", folder_level=0)
            添加路径 D:/a/b
        >>> add_path_to_sys(path="D:/a/b/c.json", folder_level=1)
            添加路径 D:/a
    """
    if path is None:
        # 默认获取该函数文件所在目录
        path = os.path.abspath(__file__)
        project_path = os.path.dirname(path)  
    else:
        if os.path.isfile(path):
            project_path = os.path.dirname(path)
        else:
            project_path = path
    
    for i in range(folder_level):
        # 向上移动一个目录层级
        project_path = os.path.dirname(project_path)
    sys.path.append(project_path)




if __name__ == '__main__':
    pass