#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1 - 工具函数模块
包含常用的工具函数和辅助方法

创建时间: 2025-07-25 21:37:58
"""

import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
import os

def save_model(model, filename):
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型
        filename: 文件名
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ 模型已保存到: {filename}")
        return True
    except Exception as e:
        print(f"✗ 模型保存失败: {str(e)}")
        return False

def load_model(filename):
    """
    从文件加载模型
    
    Args:
        filename: 模型文件名
    
    Returns:
        加载的模型
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ 模型已从 {filename} 加载")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        return None

def save_results_to_json(results, filename):
    """
    将结果保存为JSON文件
    
    Args:
        results: 结果字典
        filename: 文件名
    """
    try:
        # 处理numpy数组
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {filename}")
        return True
    except Exception as e:
        print(f"✗ 结果保存失败: {str(e)}")
        return False

def load_data_smart(file_path):
    """
    智能加载数据文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        pandas DataFrame
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # 尝试不同的编码
            for encoding in ['utf-8', 'gbk', 'gb2312']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("无法识别文件编码")
            
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
            
        elif file_ext == '.json':
            return pd.read_json(file_path)
            
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, problem_type='regression'):
    """
    计算模型评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        problem_type: 问题类型 ('regression' 或 'classification')
    
    Returns:
        指标字典
    """
    metrics = {}
    
    if problem_type == 'regression':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        
    elif problem_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted')
    
    return metrics

def normalize_data(data, method='minmax'):
    """
    数据标准化
    
    Args:
        data: 要标准化的数据
        method: 标准化方法 ('minmax', 'zscore')
    
    Returns:
        标准化后的数据和标准化器
    """
    if method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'zscore':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
    else:
        scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_time_features(df, date_column):
    """
    从日期列创建时间特征
    
    Args:
        df: 数据框
        date_column: 日期列名
    
    Returns:
        添加时间特征后的数据框
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    return df

def log_experiment(experiment_name, parameters, results, log_file="experiment_log.json"):
    """
    记录实验日志
    
    Args:
        experiment_name: 实验名称
        parameters: 实验参数
        results: 实验结果
        log_file: 日志文件名
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'parameters': parameters,
        'results': results
    }
    
    # 读取现有日志
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []
    else:
        logs = []
    
    # 添加新日志
    logs.append(log_entry)
    
    # 保存日志
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"✓ 实验日志已记录到: {log_file}")
    except Exception as e:
        print(f"✗ 日志记录失败: {str(e)}")

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    打印进度条
    
    Args:
        iteration: 当前迭代次数
        total: 总迭代次数
        prefix: 前缀字符串
        suffix: 后缀字符串
        length: 进度条长度
        fill: 填充字符
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if iteration == total:
        print()

class Timer:
    """
    计时器上下文管理器
    """
    
    def __init__(self, description="操作"):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        print(f"⏱️ 开始{self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        print(f"✓ {self.description}完成，耗时: {duration.total_seconds():.2f}秒")

# 示例使用
if __name__ == "__main__":
    print(f"🔧 问题1 工具函数模块")
    print("=" * 40)
    
    # 示例：使用计时器
    with Timer("数据处理"):
        import time
        time.sleep(1)  # 模拟耗时操作
    
    print("
🎉 工具函数模块测试完成！")
