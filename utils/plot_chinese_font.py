# https://claude.ai/chat/d2314b00-9f3a-439d-b077-0720a2cac08f
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import platform, os


def check_available_cn_fonts():
    """检查系统中可用的字体"""
    all_fonts = sorted([f.name for f in fontManager.ttflist])
    cn_keywords = ['Hei', 'Song', 'Yuan', 'Kai', 'Ming', 'SimHei', 'SimSun', 'YaHei',
                   'FangSong', 'STHeiti', 'STSong', 'STFangsong', 'STKaiti', 'PingFang',
                   'WenQuanYi', 'Noto Sans CJK', 'Source Han', '黑体', '宋体', '楷体',
                   '仿宋', '雅黑']
    cn_fonts_selected = [f for f in all_fonts if any(k in f for k in cn_keywords)]
    print("系统中可能的中文字体:")
    print("--" * 50)
    for i, font in enumerate(cn_fonts_selected, start=1):
        print(f"{i}. {font}")
    print("--" * 50)
    print(f"共找到 {len(cn_fonts_selected)} 个可能的中文字体")
    print("--" * 50)
    return cn_fonts_selected


def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    system = platform.system()
    if system == 'Windows':
        font_family = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        font_family = ['Heiti TC', 'PingFang SC', 'Songti SC']
    else:  # Linux
        font_family = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']

    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 100
    sns.set_theme(style="whitegrid")
    sns.set_style(style="whitegrid", rc={"font.sans-serif": font_family})

    print(f"中文字体设置完成，当前系统: {system}")
    print(f"使用字体: {font_family}")
    return True


class Test:
    @staticmethod
    def chinese_display():
        """测试中文显示"""
        setup_chinese_font()

        # 创建数据
        xs = np.linspace(start=0, stop=2 * np.pi, num=100)
        y1 = np.sin(xs)
        y2 = np.cos(xs)

        # 创建图像
        plt.figure(figsize=(10, 6))
        plt.plot(xs, y1, label='正弦函数', linewidth=2)
        plt.plot(xs, y2, label='余弦函数', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=np.pi, color='g', linestyle='--', alpha=0.5, label='π值')
        plt.title('三角函数图表', fontsize=16)
        plt.xlabel('弧度值 (0 - 2π)', fontsize=14)
        plt.ylabel('函数值 (-1 到 1)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.annotate(text='最大值(π/2)',
                     xy=(np.pi / 2, 1),
                     xytext=(np.pi / 2 - 0.5, 1.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        plt.tight_layout()
        plt.show()
        return "测试完成"

    @staticmethod
    def ml_visualization():
        """测试机器学习评估指标可视化中的中文显示"""
        setup_chinese_font()

        # 创建淆矩阵示例数据并绘制
        classes = ['猫', '狗', '鸟', '鱼', '兔子']
        cm = np.array([
            [85, 10, 2, 0, 3],
            [12, 78, 5, 2, 3],
            [5, 8, 80, 4, 3],
            [2, 3, 5, 85, 5],
            [5, 4, 3, 3, 85]
        ])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(label='多种动物分类混淆矩阵', fontsize=16)
        plt.xlabel(xlabel='预测类别', fontsize=14)
        plt.ylabel(ylabel='真实类别', fontsize=14)
        plt.tight_layout()
        plt.show()

        # 创建ROC曲线示例数据并绘制
        fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        tpr1 = np.array([0, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0])
        tpr2 = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr1, 'b-', label='模型A (AUC = 0.85)', linewidth=2)
        plt.plot(fpr, tpr2, 'r--', label='模型B (AUC = 0.75)', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测', alpha=0.3)
        plt.title('接收者操作特征(ROC)曲线', fontsize=16)
        plt.xlabel('假阳性率', fontsize=14)
        plt.ylabel('真阳性率', fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.grid(visible=True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        # 创建特征重要性示例数据并绘制
        features = ['年龄', '性别', '收入', '教育程度', '婚姻状况', '职业类型', '信用评分']
        importance = np.array([0.25, 0.05, 0.35, 0.15, 0.07, 0.08, 0.05])
        indices = np.argsort(importance)[::-1] # 从大到小的索引
        sorted_importance = importance[indices]
        sorted_features = [features[i] for i in indices]
        plt.figure(figsize=(10, 6))
        plt.bar(x=range(len(indices)), height=sorted_importance, align='center', alpha=0.7)
        plt.xticks(range(len(indices)), sorted_features, rotation=45)
        plt.title('特征重要性分布', fontsize=16)
        plt.xlabel('特征', fontsize=14)
        plt.ylabel('重要性分数', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("检查系统字体...")
    available_chinese_fonts = check_available_cn_fonts()

    print("\n开始测试中文显示...")
    Test.chinese_display()

    print("\n开始测试机器学习可视化...")
    Test.ml_visualization()

    print("\n所有测试完成，如果没有出现方块或乱码，说明中文显示设置成功！")
