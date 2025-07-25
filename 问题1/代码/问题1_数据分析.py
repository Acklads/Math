#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1 - 数据分析模块
用于数据探索性分析和统计描述

创建时间: 2025-07-25 21:37:58
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """
    数据分析器
    """
    
    def __init__(self, data):
        self.data = data
        self.analysis_results = {}
    
    def basic_info(self):
        """
        基本信息分析
        """
        print("📊 数据基本信息")
        print("=" * 30)
        print(f"数据形状: {self.data.shape}")
        print(f"数据类型:
{self.data.dtypes}")
        print(f"
缺失值统计:
{self.data.isnull().sum()}")
        
        # 数值型变量描述性统计
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            print(f"
📈 描述性统计:")
            print(numeric_data.describe())
    
    def correlation_analysis(self):
        """
        相关性分析
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < 2:
            print("数值型变量不足，无法进行相关性分析")
            return
        
        # 计算相关系数矩阵
        corr_matrix = numeric_data.corr()
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title(f'问题1 - 变量相关性热力图')
        plt.tight_layout()
        plt.savefig(f'../可视化结果/问题1_相关性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def distribution_analysis(self):
        """
        分布分析
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("没有数值型变量可分析")
            return
        
        n_cols = min(3, len(numeric_data.columns))
        n_rows = (len(numeric_data.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_data.columns):
            if i < len(axes):
                # 直方图
                axes[i].hist(numeric_data[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col} 分布')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('频数')
                
                # 添加统计信息
                mean_val = numeric_data[col].mean()
                std_val = numeric_data[col].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'均值: {mean_val:.2f}')
                axes[i].legend()
        
        # 隐藏多余的子图
        for i in range(len(numeric_data.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'问题1 - 变量分布分析')
        plt.tight_layout()
        plt.savefig(f'../可视化结果/问题1_分布分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def outlier_detection(self):
        """
        异常值检测
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        outliers_info = {}
        
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_data[(numeric_data[col] < lower_bound) | 
                                  (numeric_data[col] > upper_bound)][col]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(numeric_data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        numeric_data.boxplot()
        plt.title(f'问题1 - 异常值检测（箱线图）')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../可视化结果/问题1_异常值检测.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outliers_info

def main():
    """
    数据分析主函数
    """
    print(f"📊 问题1 数据分析模块")
    print("=" * 40)
    
    # 这里需要加载数据
    # data = pd.read_csv('your_data.csv')  # 替换为实际数据路径
    
    # 示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.uniform(0, 100, 1000),
        'target': np.random.normal(50, 10, 1000)
    })
    
    # 创建分析器
    analyzer = DataAnalyzer(data)
    
    # 执行分析
    analyzer.basic_info()
    print("
" + "="*50 + "
")
    
    corr_matrix = analyzer.correlation_analysis()
    print("
" + "="*50 + "
")
    
    analyzer.distribution_analysis()
    print("
" + "="*50 + "
")
    
    outliers_info = analyzer.outlier_detection()
    
    print("
🎉 数据分析完成！")

if __name__ == "__main__":
    main()
