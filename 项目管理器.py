import os
import json
from datetime import datetime

class MathModelingProjectManager:
    """
    数学建模项目管理器
    用于创建和管理数学建模竞赛项目结构
    """
    
    def __init__(self, project_root="."):
        self.project_root = project_root
        self.problems = ["问题1", "问题2", "问题3"]
        
    def create_problem_templates(self):
        """
        为每个问题创建代码模板文件
        """
        print("🔧 正在创建问题代码模板...")
        
        for problem in self.problems:
            problem_path = os.path.join(self.project_root, problem)
            code_path = os.path.join(problem_path, "代码")
            
            if os.path.exists(code_path):
                # 创建主要代码文件
                self._create_main_code_file(code_path, problem)
                self._create_data_analysis_file(code_path, problem)
                self._create_visualization_file(code_path, problem)
                self._create_utils_file(code_path, problem)
                
                print(f"✓ {problem} 模板文件创建完成")
            else:
                print(f"✗ {problem} 文件夹不存在")
    
    def _create_main_code_file(self, code_path, problem):
        """
        创建主代码文件
        """
        filename = os.path.join(code_path, f"{problem}_主程序.py")
        
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{problem} - 主程序
数学建模竞赛解决方案

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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

class {problem.replace('问题', 'Problem')}Solver:
    """
    {problem}求解器
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {{}}
        
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
                    
                print(f"✓ 数据加载成功，形状: {{self.data.shape}}")
                return True
            else:
                # 生成示例数据
                print("⚠️ 未指定数据路径，生成示例数据")
                self.data = self._generate_sample_data()
                return True
                
        except Exception as e:
            print(f"✗ 数据加载失败: {{str(e)}}")
            return False
    
    def _generate_sample_data(self):
        """
        生成示例数据
        """
        np.random.seed(42)
        n_samples = 100
        
        data = {{
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples),
            'x3': np.random.normal(0, 1, n_samples),
        }}
        
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
            print(f"发现缺失值:\n{{missing_values[missing_values > 0]}}")
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
                print(f"{{col}} 列发现 {{outliers}} 个异常值")
        
        print("✓ 数据预处理完成")
        return True
    
    def solve(self):
        """
        求解主函数
        """
        print(f"🎯 开始求解{problem}...")
        
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
            
            self.results = {{
                'mse': mse,
                'r2': r2,
                'coefficients': self.model.coef_.tolist(),
                'intercept': self.model.intercept_
            }}
            
            print(f"✓ 求解完成")
            print(f"  均方误差 (MSE): {{mse:.4f}}")
            print(f"  决定系数 (R²): {{r2:.4f}}")
            
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
            result_file = os.path.join(output_dir, f"{problem}_结果.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 结果已保存到: {{result_file}}")
            return True
            
        except Exception as e:
            print(f"✗ 保存结果失败: {{str(e)}}")
            return False

def main():
    """
    主函数
    """
    print(f"🚀 {problem} 求解程序启动")
    print("=" * 50)
    
    # 创建求解器
    solver = {problem.replace('问题', 'Problem')}Solver()
    
    # 执行求解流程
    if solver.load_data():
        if solver.data_preprocessing():
            if solver.solve():
                solver.save_results()
                print(f"\n🎉 {problem} 求解完成！")
            else:
                print(f"\n❌ {problem} 求解失败")
        else:
            print(f"\n❌ 数据预处理失败")
    else:
        print(f"\n❌ 数据加载失败")

if __name__ == "__main__":
    main()
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_data_analysis_file(self, code_path, problem):
        """
        创建数据分析文件
        """
        filename = os.path.join(code_path, f"{problem}_数据分析.py")
        
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{problem} - 数据分析模块
用于数据探索性分析和统计描述

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
        self.analysis_results = {{}}
    
    def basic_info(self):
        """
        基本信息分析
        """
        print("📊 数据基本信息")
        print("=" * 30)
        print(f"数据形状: {{self.data.shape}}")
        print(f"数据类型:\n{{self.data.dtypes}}")
        print(f"\n缺失值统计:\n{{self.data.isnull().sum()}}")
        
        # 数值型变量描述性统计
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            print(f"\n📈 描述性统计:")
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
        plt.title(f'{problem} - 变量相关性热力图')
        plt.tight_layout()
        plt.savefig(f'../可视化结果/{problem}_相关性分析.png', dpi=300, bbox_inches='tight')
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
                axes[i].set_title(f'{{col}} 分布')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('频数')
                
                # 添加统计信息
                mean_val = numeric_data[col].mean()
                std_val = numeric_data[col].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'均值: {{mean_val:.2f}}')
                axes[i].legend()
        
        # 隐藏多余的子图
        for i in range(len(numeric_data.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{problem} - 变量分布分析')
        plt.tight_layout()
        plt.savefig(f'../可视化结果/{problem}_分布分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def outlier_detection(self):
        """
        异常值检测
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        outliers_info = {{}}
        
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_data[(numeric_data[col] < lower_bound) | 
                                  (numeric_data[col] > upper_bound)][col]
            
            outliers_info[col] = {{
                'count': len(outliers),
                'percentage': len(outliers) / len(numeric_data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }}
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        numeric_data.boxplot()
        plt.title(f'{problem} - 异常值检测（箱线图）')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../可视化结果/{problem}_异常值检测.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outliers_info

def main():
    """
    数据分析主函数
    """
    print(f"📊 {problem} 数据分析模块")
    print("=" * 40)
    
    # 这里需要加载数据
    # data = pd.read_csv('your_data.csv')  # 替换为实际数据路径
    
    # 示例数据
    np.random.seed(42)
    data = pd.DataFrame({{
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.uniform(0, 100, 1000),
        'target': np.random.normal(50, 10, 1000)
    }})
    
    # 创建分析器
    analyzer = DataAnalyzer(data)
    
    # 执行分析
    analyzer.basic_info()
    print("\n" + "="*50 + "\n")
    
    corr_matrix = analyzer.correlation_analysis()
    print("\n" + "="*50 + "\n")
    
    analyzer.distribution_analysis()
    print("\n" + "="*50 + "\n")
    
    outliers_info = analyzer.outlier_detection()
    
    print("\n🎉 数据分析完成！")

if __name__ == "__main__":
    main()
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_visualization_file(self, code_path, problem):
        """
        创建可视化文件
        """
        filename = os.path.join(code_path, f"{problem}_可视化.py")
        
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{problem} - 可视化模块
用于生成各种图表和可视化结果

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
            axes[1,1].set_title(f'{{cols[0]}} vs {{cols[1]}}')
        
        plt.suptitle(f'{problem} - 数据概览', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{{self.output_dir}}/{problem}_数据概览.png', 
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
        
        plt.suptitle(f'{problem} - 结果可视化')
        plt.tight_layout()
        plt.savefig(f'{{self.output_dir}}/{problem}_结果可视化.png', 
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
            title=f'{problem} - 交互式散点图矩阵',
            height=600
        )
        
        # 保存为HTML文件
        fig.write_html(f'{{self.output_dir}}/{problem}_交互式图表.html')
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
                   f'{{value:.4f}}', ha='center', va='bottom')
        
        ax.set_title(f'{problem} - 模型性能指标')
        ax.set_ylabel('指标值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{{self.output_dir}}/{problem}_模型性能.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report_plots(self):
        """
        生成报告用图表
        """
        print(f"🎨 正在生成{problem}可视化报告...")
        
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
            print(f"交互式图表生成失败: {{e}}")
        
        print(f"✓ {problem}可视化报告生成完成")
        print(f"📁 图表保存位置: {{self.output_dir}}")

def main():
    """
    可视化主函数
    """
    print(f"🎨 {problem} 可视化模块")
    print("=" * 40)
    
    # 示例数据
    np.random.seed(42)
    data = pd.DataFrame({{
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(2, 200),
        'feature3': np.random.uniform(0, 100, 200),
        'target': np.random.normal(50, 10, 200)
    }})
    
    # 示例结果
    results = {{
        'metrics': {{
            'MSE': 0.1234,
            'R²': 0.8765,
            'MAE': 0.0987
        }}
    }}
    
    # 创建可视化器
    visualizer = Visualizer(data, results)
    
    # 生成报告图表
    visualizer.generate_report_plots()
    
    print("\n🎉 可视化完成！")

if __name__ == "__main__":
    main()
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_utils_file(self, code_path, problem):
        """
        创建工具函数文件
        """
        filename = os.path.join(code_path, f"{problem}_工具函数.py")
        
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{problem} - 工具函数模块
包含常用的工具函数和辅助方法

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
        print(f"✓ 模型已保存到: {{filename}}")
        return True
    except Exception as e:
        print(f"✗ 模型保存失败: {{str(e)}}")
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
        print(f"✓ 模型已从 {{filename}} 加载")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {{str(e)}}")
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
                return {{k: convert_numpy(v) for k, v in obj.items()}}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {{filename}}")
        return True
    except Exception as e:
        print(f"✗ 结果保存失败: {{str(e)}}")
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
            raise ValueError(f"不支持的文件格式: {{file_ext}}")
            
    except Exception as e:
        print(f"✗ 数据加载失败: {{str(e)}}")
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
    metrics = {{}}
    
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
        raise ValueError(f"不支持的标准化方法: {{method}}")
    
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
    log_entry = {{
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'parameters': parameters,
        'results': results
    }}
    
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
        print(f"✓ 实验日志已记录到: {{log_file}}")
    except Exception as e:
        print(f"✗ 日志记录失败: {{str(e)}}")

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
    percent = ("{{0:.1f}}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{{prefix}} |{{bar}}| {{percent}}% {{suffix}}', end='\r')
    
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
        print(f"⏱️ 开始{{self.description}}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        print(f"✓ {{self.description}}完成，耗时: {{duration.total_seconds():.2f}}秒")

# 示例使用
if __name__ == "__main__":
    print(f"🔧 {problem} 工具函数模块")
    print("=" * 40)
    
    # 示例：使用计时器
    with Timer("数据处理"):
        import time
        time.sleep(1)  # 模拟耗时操作
    
    print("\n🎉 工具函数模块测试完成！")
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def create_readme_files(self):
        """
        为每个问题文件夹创建README文件
        """
        print("📝 正在创建README文件...")
        
        for problem in self.problems:
            readme_path = os.path.join(self.project_root, problem, "README.md")
            
            content = f'''# {problem}

## 问题描述
请在此处填写{problem}的具体描述和要求。

## 解决思路
1. 数据分析和预处理
2. 模型建立
3. 求解和优化
4. 结果验证和可视化

## 文件结构
```
{problem}/
├── 代码/
│   ├── {problem}_主程序.py          # 主要求解程序
│   ├── {problem}_数据分析.py        # 数据分析模块
│   ├── {problem}_可视化.py          # 可视化模块
│   └── {problem}_工具函数.py        # 工具函数模块
├── 可视化结果/                      # 图表和结果文件
└── README.md                        # 本文件
```

## 使用说明
1. 首先运行数据分析模块，了解数据特征
2. 根据分析结果调整主程序中的算法
3. 运行主程序进行求解
4. 使用可视化模块生成图表

## 依赖库
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- plotly (可选，用于交互式图表)

## 注意事项
- 请根据实际题目要求调整代码
- 确保数据文件路径正确
- 运行前检查所需库是否已安装

## 更新日志
- {datetime.now().strftime('%Y-%m-%d')}: 创建项目结构
'''
            
            try:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ {problem}/README.md 创建完成")
            except Exception as e:
                print(f"✗ {problem}/README.md 创建失败: {str(e)}")
    
    def create_main_readme(self):
        """
        创建项目主README文件
        """
        readme_path = os.path.join(self.project_root, "README.md")
        
        content = f'''# 数学建模竞赛项目

## 项目概述
本项目为全国数学建模竞赛准备的标准化项目结构，包含完整的代码模板和工具。

## 项目结构
```
Math_test_project/
├── 问题1/
│   ├── 代码/
│   │   ├── 问题1_主程序.py
│   │   ├── 问题1_数据分析.py
│   │   ├── 问题1_可视化.py
│   │   └── 问题1_工具函数.py
│   ├── 可视化结果/
│   └── README.md
├── 问题2/
│   ├── 代码/
│   ├── 可视化结果/
│   └── README.md
├── 问题3/
│   ├── 代码/
│   ├── 可视化结果/
│   └── README.md
├── 题目分析器.py                    # Word文档题目分析工具
├── 项目管理器.py                    # 项目管理工具
├── read_docx.py                     # Word文档读取工具
└── README.md                        # 本文件
```

## 快速开始

### 1. 环境准备
```bash
# 安装必要的Python库
python -m pip install numpy pandas matplotlib seaborn scikit-learn python-docx plotly
```

### 2. 题目分析
```bash
# 将比赛题目Word文档放入项目根目录
# 运行题目分析器
python 题目分析器.py
```

### 3. 开始解题
1. 查看生成的题目分析报告
2. 进入对应问题文件夹
3. 根据题目要求修改代码模板
4. 运行求解程序

## 工具说明

### 题目分析器 (题目分析器.py)
- 自动读取Word格式的比赛题目
- 识别问题结构和关键词
- 生成分析报告和解题建议
- 输出JSON格式的分析结果

### 项目管理器 (项目管理器.py)
- 创建标准化的项目结构
- 生成代码模板文件
- 管理项目文件和目录

### 代码模板特性
- **主程序**: 完整的求解流程框架
- **数据分析**: 数据探索和预处理工具
- **可视化**: 图表生成和结果展示
- **工具函数**: 常用的辅助函数

## 使用建议

### 比赛流程
1. **题目理解** (30分钟)
   - 使用题目分析器快速理解题目
   - 识别关键问题和约束条件
   
2. **数据分析** (1-2小时)
   - 运行数据分析模块
   - 理解数据特征和分布
   
3. **模型建立** (4-6小时)
   - 根据问题特点选择合适算法
   - 修改主程序模板
   
4. **求解优化** (2-4小时)
   - 调试和优化算法
   - 验证结果合理性
   
5. **结果展示** (1-2小时)
   - 生成可视化图表
   - 整理最终结果

### 团队协作
- 每个成员负责一个问题文件夹
- 使用统一的代码风格和注释
- 定期同步进度和结果

## 常用算法库

### 优化算法
- scipy.optimize: 数学优化
- cvxpy: 凸优化
- pulp: 线性规划

### 机器学习
- scikit-learn: 经典机器学习
- tensorflow/pytorch: 深度学习
- xgboost: 梯度提升

### 数值计算
- numpy: 数值计算基础
- scipy: 科学计算
- sympy: 符号计算

## 注意事项
- 确保所有代码都有中文注释
- 保存好中间结果和模型文件
- 及时备份重要代码和数据
- 注意时间管理，合理分配各问题时间

## 更新日志
- {datetime.now().strftime('%Y-%m-%d')}: 创建项目结构和模板

---

**祝比赛顺利！🏆**
'''
        
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✓ 主README.md 创建完成")
        except Exception as e:
            print(f"✗ 主README.md 创建失败: {str(e)}")
    
    def install_dependencies(self):
        """
        安装项目依赖
        """
        print("📦 正在安装项目依赖...")
        
        dependencies = [
            'numpy',
            'pandas', 
            'matplotlib',
            'seaborn',
            'scikit-learn',
            'plotly',
            'scipy'
        ]
        
        for dep in dependencies:
            print(f"安装 {dep}...")
            # 这里只是示例，实际安装需要在命令行执行
        
        print("💡 请手动运行以下命令安装依赖:")
        print(f"python -m pip install {' '.join(dependencies)}")
    
    def show_project_status(self):
        """
        显示项目状态
        """
        print("\n📊 项目状态概览")
        print("=" * 50)
        
        for problem in self.problems:
            problem_path = os.path.join(self.project_root, problem)
            if os.path.exists(problem_path):
                code_path = os.path.join(problem_path, "代码")
                result_path = os.path.join(problem_path, "可视化结果")
                
                code_files = len([f for f in os.listdir(code_path) 
                                if f.endswith('.py')]) if os.path.exists(code_path) else 0
                result_files = len(os.listdir(result_path)) if os.path.exists(result_path) else 0
                
                print(f"📁 {problem}:")
                print(f"   代码文件: {code_files} 个")
                print(f"   结果文件: {result_files} 个")
                print(f"   状态: {'✓ 已创建' if code_files > 0 else '⚠️ 待开发'}")
            else:
                print(f"❌ {problem}: 文件夹不存在")

def main():
    """
    主函数
    """
    print("🎯 数学建模项目管理器")
    print("=" * 40)
    
    manager = MathModelingProjectManager()
    
    print("\n1. 创建代码模板...")
    manager.create_problem_templates()
    
    print("\n2. 创建README文件...")
    manager.create_readme_files()
    manager.create_main_readme()
    
    print("\n3. 显示项目状态...")
    manager.show_project_status()
    
    print("\n4. 依赖安装提示...")
    manager.install_dependencies()
    
    print("\n🎉 项目初始化完成！")
    print("\n📋 接下来的步骤:")
    print("1. 将比赛题目Word文档放入项目根目录")
    print("2. 运行 'python 题目分析器.py' 分析题目")
    print("3. 根据分析结果开始在各问题文件夹中编码")
    print("4. 使用可视化模块生成图表和结果")

if __name__ == "__main__":
    main()