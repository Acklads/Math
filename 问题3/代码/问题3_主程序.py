#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3 - 主程序
数学建模竞赛解决方案

创建时间: 2025-07-25 21:37:58
作者: [请填写团队信息]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem3Solver:
    """
    问题3求解器
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {}
        
    def load_data(self, data_path=None):
        """
        加载数据
        """
        try:
            if data_path:
                # 根据文件类型加载数据
                if data_path.endswith('.csv'):
                    self.data = pd.read_csv(data_path, encoding='utf-8')
                elif data_path.endswith('.xlsx'):
                    self.data = pd.read_excel(data_path)
                else:
                    print("不支持的文件格式")
                    return False
                    
                print(f"✓ 数据加载成功，形状: {self.data.shape}")
                return True
            else:
                # 生成示例数据
                print("⚠️ 未指定数据路径，生成示例数据")
                self.data = self._generate_sample_data()
                return True
                
        except Exception as e:
            print(f"✗ 数据加载失败: {str(e)}")
            return False
    
    def _generate_sample_data(self):
        """
        生成示例数据
        """
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples),
            'x3': np.random.normal(0, 1, n_samples),
        }
        
        # 生成目标变量
        data['y'] = (2 * data['x1'] + 3 * data['x2'] + 
                    np.random.normal(0, 0.1, n_samples))
        
        return pd.DataFrame(data)
    
    def data_preprocessing(self):
        """
        数据预处理
        """
        if self.data is None:
            print("✗ 请先加载数据")
            return False
        
        print("🔄 开始数据预处理...")
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print(f"发现缺失值:
{missing_values[missing_values > 0]}")
            # 处理缺失值（这里使用均值填充，实际应根据具体情况调整）
            self.data = self.data.fillna(self.data.mean())
        
        # 检查异常值
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | 
                       (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                print(f"{col} 列发现 {outliers} 个异常值")
        
        print("✓ 数据预处理完成")
        return True
    
    def solve(self):
        """
        求解主函数
        """
        print(f"🎯 开始求解问题3...")
        
        # TODO: 在这里实现具体的求解算法
        # 示例：简单的线性回归
        if self.data is not None and 'y' in self.data.columns:
            X = self.data.drop('y', axis=1)
            y = self.data['y']
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 简单的线性回归示例
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # 预测
            y_pred = self.model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results = {
                'mse': mse,
                'r2': r2,
                'coefficients': self.model.coef_.tolist(),
                'intercept': self.model.intercept_
            }
            
            print(f"✓ 求解完成")
            print(f"  均方误差 (MSE): {mse:.4f}")
            print(f"  决定系数 (R²): {r2:.4f}")
            
            return True
        else:
            print("✗ 数据格式不正确或缺少目标变量")
            return False
    
    def save_results(self, output_dir="../可视化结果"):
        """
        保存结果
        """
        if not self.results:
            print("✗ 没有结果可保存")
            return False
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果到JSON文件
            result_file = os.path.join(output_dir, f"问题3_结果.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 结果已保存到: {result_file}")
            return True
            
        except Exception as e:
            print(f"✗ 保存结果失败: {str(e)}")
            return False

def main():
    """
    主函数
    """
    print(f"🚀 问题3 求解程序启动")
    print("=" * 50)
    
    # 创建求解器
    solver = Problem3Solver()
    
    # 执行求解流程
    if solver.load_data():
        if solver.data_preprocessing():
            if solver.solve():
                solver.save_results()
                print(f"
🎉 问题3 求解完成！")
            else:
                print(f"
❌ 问题3 求解失败")
        else:
            print(f"
❌ 数据预处理失败")
    else:
        print(f"
❌ 数据加载失败")

if __name__ == "__main__":
    main()
