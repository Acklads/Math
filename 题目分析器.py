from docx import Document
import re
import os
import json
from datetime import datetime

class MathModelingAnalyzer:
    """
    数学建模竞赛题目分析器
    用于读取Word文档中的题目内容，分析题目结构，并生成项目管理信息
    """
    
    def __init__(self, docx_path):
        self.docx_path = docx_path
        self.content = ""
        self.problems = {}
        self.analysis_result = {}
        
    def read_document(self):
        """
        读取Word文档内容
        """
        try:
            doc = Document(self.docx_path)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            self.content = '\n'.join(full_text)
            print(f"✓ 成功读取文档: {os.path.basename(self.docx_path)}")
            return True
            
        except Exception as e:
            print(f"✗ 读取文档失败: {str(e)}")
            return False
    
    def analyze_problems(self):
        """
        分析题目中的问题结构
        """
        if not self.content:
            print("✗ 请先读取文档内容")
            return False
        
        # 查找问题标识
        problem_patterns = [
            r'问题[一二三四五六七八九十1-9][:：]?',
            r'Problem\s*[1-9][:：]?',
            r'第[一二三四五六七八九十1-9][个题]问题[:：]?',
            r'\([1-9]\)',
            r'[1-9]\.',
        ]
        
        lines = self.content.split('\n')
        current_problem = None
        
        for i, line in enumerate(lines):
            # 检查是否是问题标题
            for pattern in problem_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # 提取问题编号
                    problem_num = self._extract_problem_number(line)
                    if problem_num and problem_num <= 3:  # 通常数学建模有3个问题
                        current_problem = problem_num
                        self.problems[current_problem] = {
                            'title': line,
                            'content': [],
                            'line_start': i
                        }
                        print(f"✓ 发现问题{current_problem}: {line[:50]}...")
                    break
            else:
                # 如果当前行不是问题标题，且有当前问题，则添加到内容中
                if current_problem and line.strip():
                    self.problems[current_problem]['content'].append(line)
        
        return len(self.problems) > 0
    
    def _extract_problem_number(self, text):
        """
        从文本中提取问题编号
        """
        # 中文数字映射
        chinese_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
        
        # 查找阿拉伯数字
        arabic_match = re.search(r'[1-9]', text)
        if arabic_match:
            return int(arabic_match.group())
        
        # 查找中文数字
        for chinese, num in chinese_nums.items():
            if chinese in text:
                return num
        
        return None
    
    def generate_analysis_report(self):
        """
        生成分析报告
        """
        self.analysis_result = {
            'document_info': {
                'filename': os.path.basename(self.docx_path),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_lines': len(self.content.split('\n')),
                'total_characters': len(self.content)
            },
            'problems_found': len(self.problems),
            'problems_detail': {}
        }
        
        for prob_num, prob_data in self.problems.items():
            content_text = '\n'.join(prob_data['content'])
            self.analysis_result['problems_detail'][f'问题{prob_num}'] = {
                'title': prob_data['title'],
                'content_length': len(content_text),
                'content_lines': len(prob_data['content']),
                'keywords': self._extract_keywords(content_text),
                'suggested_methods': self._suggest_methods(content_text)
            }
        
        return self.analysis_result
    
    def _extract_keywords(self, text):
        """
        提取关键词
        """
        # 数学建模常见关键词
        keywords = [
            '优化', '建模', '预测', '分析', '算法', '模型', '数据', '统计',
            '回归', '聚类', '分类', '神经网络', '机器学习', '深度学习',
            '线性规划', '非线性', '动态规划', '遗传算法', '模拟退火',
            '蒙特卡洛', '时间序列', '概率', '随机', '仿真', '评价',
            '决策', '多目标', '约束', '最优化', '求解'
        ]
        
        found_keywords = []
        for keyword in keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # 返回前10个关键词
    
    def _suggest_methods(self, text):
        """
        根据内容建议可能的解决方法
        """
        suggestions = []
        
        if any(word in text for word in ['优化', '最优', '最大', '最小']):
            suggestions.append('数学优化方法')
        
        if any(word in text for word in ['预测', '预报', '趋势']):
            suggestions.append('时间序列分析')
        
        if any(word in text for word in ['分类', '聚类', '识别']):
            suggestions.append('机器学习方法')
        
        if any(word in text for word in ['评价', '评估', '排序']):
            suggestions.append('多指标评价方法')
        
        if any(word in text for word in ['网络', '图', '路径']):
            suggestions.append('图论方法')
        
        if any(word in text for word in ['仿真', '模拟', '随机']):
            suggestions.append('蒙特卡洛仿真')
        
        return suggestions if suggestions else ['需要进一步分析']
    
    def save_analysis(self, output_file='题目分析结果.json'):
        """
        保存分析结果到JSON文件
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_result, f, ensure_ascii=False, indent=2)
            print(f"✓ 分析结果已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return False
    
    def print_summary(self):
        """
        打印分析摘要
        """
        print("\n" + "="*60)
        print("📊 数学建模题目分析报告")
        print("="*60)
        
        if self.analysis_result:
            doc_info = self.analysis_result['document_info']
            print(f"📄 文档名称: {doc_info['filename']}")
            print(f"⏰ 分析时间: {doc_info['analysis_time']}")
            print(f"📝 总行数: {doc_info['total_lines']}")
            print(f"🔤 总字符数: {doc_info['total_characters']}")
            print(f"❓ 发现问题数: {self.analysis_result['problems_found']}")
            
            print("\n📋 问题详情:")
            for prob_name, prob_detail in self.analysis_result['problems_detail'].items():
                print(f"\n  {prob_name}:")
                print(f"    标题: {prob_detail['title']}")
                print(f"    内容长度: {prob_detail['content_length']} 字符")
                print(f"    关键词: {', '.join(prob_detail['keywords'])}")
                print(f"    建议方法: {', '.join(prob_detail['suggested_methods'])}")
        
        print("\n" + "="*60)

def main():
    """
    主函数
    """
    print("🎯 数学建模竞赛题目分析器")
    print("=" * 40)
    
    # 查找项目中的Word文档
    docx_files = [f for f in os.listdir('.') if f.endswith('.docx')]
    
    if not docx_files:
        print("❌ 未找到Word文档文件")
        print("请将题目Word文档放在项目根目录下")
        return
    
    # 如果有多个文档，选择第一个或让用户选择
    if len(docx_files) == 1:
        selected_file = docx_files[0]
        print(f"📄 找到文档: {selected_file}")
    else:
        print("📄 找到多个Word文档:")
        for i, file in enumerate(docx_files, 1):
            print(f"  {i}. {file}")
        
        try:
            choice = int(input("请选择要分析的文档编号: ")) - 1
            selected_file = docx_files[choice]
        except (ValueError, IndexError):
            print("❌ 无效选择，使用第一个文档")
            selected_file = docx_files[0]
    
    # 创建分析器并执行分析
    analyzer = MathModelingAnalyzer(selected_file)
    
    if analyzer.read_document():
        if analyzer.analyze_problems():
            analyzer.generate_analysis_report()
            analyzer.print_summary()
            analyzer.save_analysis()
            
            print("\n🎉 分析完成！")
            print("💡 建议接下来的步骤:")
            print("  1. 查看生成的分析报告文件")
            print("  2. 根据问题特点选择合适的算法和工具")
            print("  3. 在对应的问题文件夹中开始编码")
            print("  4. 将可视化结果保存到对应的结果文件夹")
        else:
            print("❌ 未能识别出问题结构，请检查文档格式")
    else:
        print("❌ 文档读取失败")

if __name__ == "__main__":
    main()