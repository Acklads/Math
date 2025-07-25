#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2 - 可视化模块
用于生成各种图表和可视化结果

创建时间: 2025-07-25 21:37:58
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class Visualizer:
    """
    可视化器
    """
    
    def __init__(self, data, results=None):
        self.data = data
        self.results = results
        self.output_dir = "../可视化结果"
        
        # 确保输出目录存在
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_data_overview(self):
        """
        数据概览图
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("没有数值型数据可可视化")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 数据分布概览
        numeric_data.hist(bins=30, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('数据分布概览')
        
        # 2. 相关性热力图
        if numeric_data.shape[1] > 1:
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0,1])
            axes[0,1].set_title('相关性矩阵')
        
        # 3. 箱线图
        numeric_data.boxplot(ax=axes[1,0])
        axes[1,0].set_title('异常值检测')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 散点图矩阵（如果变量不多）
        if numeric_data.shape[1] <= 4:
            pd.plotting.scatter_matrix(numeric_data, ax=axes[1,1], alpha=0.6)
            axes[1,1].set_title('散点图矩阵')
        else:
            # 选择前两个变量绘制散点图
            cols = numeric_data.columns[:2]
            axes[1,1].scatter(numeric_data[cols[0]], numeric_data[cols[1]], alpha=0.6)
            axes[1,1].set_xlabel(cols[0])
            axes[1,1].set_ylabel(cols[1])
            axes[1,1].set_title(f'{cols[0]} vs {cols[1]}')
        
        plt.suptitle(f'问题2 - 数据概览', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/问题2_数据概览.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self):
        """
        结果可视化
        """
        if not self.results:
            print("没有结果数据可可视化")
            return
        
        # 根据结果类型创建不同的图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 示例：如果有预测结果
        if 'predictions' in self.results:
            # 实际值 vs 预测值
            actual = self.results.get('actual', [])
            predicted = self.results.get('predictions', [])
            
            if actual and predicted:
                axes[0].scatter(actual, predicted, alpha=0.6)
                axes[0].plot([min(actual), max(actual)], 
                           [min(actual), max(actual)], 'r--', lw=2)
                axes[0].set_xlabel('实际值')
                axes[0].set_ylabel('预测值')
                axes[0].set_title('实际值 vs 预测值')
                
                # 残差图
                residuals = np.array(actual) - np.array(predicted)
                axes[1].scatter(predicted, residuals, alpha=0.6)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_xlabel('预测值')
                axes[1].set_ylabel('残差')
                axes[1].set_title('残差图')
        
        # 如果有其他类型的结果，可以在这里添加
        
        plt.suptitle(f'问题2 - 结果可视化')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/问题2_结果可视化.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plot(self):
        """
        创建交互式图表
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("数据维度不足，无法创建交互式图表")
            return
        
        # 创建交互式散点图
        fig = px.scatter_matrix(
            numeric_data,
            title=f'问题2 - 交互式散点图矩阵',
            height=600
        )
        
        # 保存为HTML文件
        fig.write_html(f'{self.output_dir}/问题2_交互式图表.html')
        fig.show()
    
    def plot_model_performance(self, metrics):
        """
        模型性能可视化
        """
        if not metrics:
            return
        
        # 创建性能指标图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, 
                     color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        ax.set_title(f'问题2 - 模型性能指标')
        ax.set_ylabel('指标值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/问题2_模型性能.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report_plots(self):
        """
        生成报告用图表
        """
        print(f"🎨 正在生成问题2可视化报告...")
        
        # 生成所有图表
        self.plot_data_overview()
        
        if self.results:
            self.plot_results()
            
            # 如果有性能指标
            if 'metrics' in self.results:
                self.plot_model_performance(self.results['metrics'])
        
        # 生成交互式图表
        try:
            self.create_interactive_plot()
        except Exception as e:
            print(f"交互式图表生成失败: {e}")
        
        print(f"✓ 问题2可视化报告生成完成")
        print(f"📁 图表保存位置: {self.output_dir}")

def main():
    """
    可视化主函数
    """
    print(f"🎨 问题2 可视化模块")
    print("=" * 40)
    
    # 示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(2, 200),
        'feature3': np.random.uniform(0, 100, 200),
        'target': np.random.normal(50, 10, 200)
    })
    
    # 示例结果
    results = {
        'metrics': {
            'MSE': 0.1234,
            'R²': 0.8765,
            'MAE': 0.0987
        }
    }
    
    # 创建可视化器
    visualizer = Visualizer(data, results)
    
    # 生成报告图表
    visualizer.generate_report_plots()
    
    print("
🎉 可视化完成！")

if __name__ == "__main__":
    main()
