import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import itertools


class MLEvaluationVisualization:
    """
    机器学习评估与可视化模块
    用于输出模型指标（Accuracy、Precision、Recall、F1-score），
    并绘制混淆矩阵与特征分布图，强化模型理解与诊断能力。
    """

    def __init__(self, model, X_train, X_test, y_train, y_test, class_names=None, feature_names=None):
        """
        初始化评估与可视化模块

        参数:
            model: 训练好的机器学习模型
            X_train: 训练特征数据
            X_test: 测试特征数据
            y_train: 训练标签数据
            y_test: 测试标签数据
            class_names: 类别名称列表（可选）
            feature_names: 特征名称列表（可选）
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

        # 设置类别名称和特征名称
        self.class_names = class_names
        self.feature_names = feature_names

        # 如果没有提供类别名称，根据唯一标签值创建
        if self.class_names is None:
            unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
            self.class_names = [f"类别 {i}" for i in unique_labels]

        # 如果没有提供特征名称，创建默认特征名称
        if self.feature_names is None and X_train is not None:
            self.feature_names = [f"特征 {i}" for i in range(X_train.shape[1])]

    def calculate_metrics(self):
        """计算并返回评估指标"""
        metrics = {}

        # 基本分类指标
        metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)

        # 处理二分类和多分类情况
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        if len(unique_labels) == 2:
            metrics['precision'] = precision_score(self.y_test, self.y_pred)
            metrics['recall'] = recall_score(self.y_test, self.y_pred)
            metrics['f1'] = f1_score(self.y_test, self.y_pred)
        else:
            # 多分类使用 macro 平均
            metrics['precision'] = precision_score(self.y_test, self.y_pred, average='macro')
            metrics['recall'] = recall_score(self.y_test, self.y_pred, average='macro')
            metrics['f1'] = f1_score(self.y_test, self.y_pred, average='macro')

        return metrics

    def print_metrics(self):
        """打印评估指标"""
        metrics = self.calculate_metrics()

        print("=" * 50)
        print("模型评估指标:")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数 (F1-Score): {metrics['f1']:.4f}")
        print("=" * 50)

        # 打印详细的分类报告
        print("\n分类报告:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        return metrics

    def plot_confusion_matrix(self, normalize=False, figsize=(10, 8), cmap=plt.cm.Blues, title=None):
        """
        绘制混淆矩阵

        参数:
            normalize: 是否归一化混淆矩阵
            figsize: 图形大小
            cmap: 颜色映射
            title: 图表标题（可选）
        """
        cm = confusion_matrix(self.y_test, self.y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            if title is None:
                title = '归一化混淆矩阵'
        else:
            fmt = 'd'
            if title is None:
                title = '混淆矩阵'

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title(title)
        plt.tight_layout()
        plt.show()

        return cm

    def plot_feature_importance(self, kind='model', n_features=10, figsize=(12, 6)):
        """
        绘制特征重要性图

        参数:
            kind: 特征重要性类型 ('model' 或 'permutation')
            n_features: 要显示的特征数量
            figsize: 图形大小
        """
        if self.feature_names is None:
            raise ValueError("需要提供特征名称才能绘制特征重要性图")

        plt.figure(figsize=figsize)

        # 基于模型的特征重要性
        if kind == 'model':
            # 检查模型是否有 feature_importances_ 属性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                raise ValueError("所选模型没有 feature_importances_ 属性，请使用 'permutation' 类型")

        # 基于排列的特征重要性
        elif kind == 'permutation':
            result = permutation_importance(self.model, self.X_test, self.y_test,
                                            n_repeats=10, random_state=42)
            importances = result.importances_mean

        else:
            raise ValueError("kind 参数必须是 'model' 或 'permutation'")

        # 获取前 n 个特征
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:n_features]

        # 绘制条形图
        plt.barh(range(len(top_indices)), importances[top_indices])
        plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
        plt.xlabel('特征重要性')
        plt.ylabel('特征')

        if kind == 'model':
            plt.title('基于模型的特征重要性')
        else:
            plt.title('基于排列的特征重要性')

        plt.tight_layout()
        plt.show()

        return importances

    def plot_feature_distributions(self, features=None, n_cols=3, figsize=(15, 10)):
        """
        绘制特征分布图

        参数:
            features: 要绘制的特征索引或名称列表 (可选)
            n_cols: 每行的子图数量
            figsize: 图形大小
        """
        # 如果未指定特征，使用所有特征
        if features is None:
            feature_indices = range(self.X_train.shape[1])
            feature_names = self.feature_names
        else:
            # 处理特征名称或索引
            if isinstance(features[0], str):
                feature_indices = [self.feature_names.index(f) for f in features]
                feature_names = features
            else:
                feature_indices = features
                feature_names = [self.feature_names[i] for i in feature_indices]

        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
            ax = axes[i]

            # 为每个类别绘制密度图
            for j, label in enumerate(unique_labels):
                train_mask = self.y_train == label
                test_mask = self.y_test == label

                # 训练集
                sns.kdeplot(self.X_train[train_mask, idx], ax=ax, color=colors[j],
                            label=f"{self.class_names[j]} (训练)", linestyle='-')

                # 测试集
                sns.kdeplot(self.X_test[test_mask, idx], ax=ax, color=colors[j],
                            label=f"{self.class_names[j]} (测试)", linestyle='--')

            ax.set_title(name)
            ax.set_xlabel('特征值')
            ax.set_ylabel('密度')

            # 只在第一个子图显示图例
            if i == 0:
                ax.legend()

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, figsize=(10, 8)):
        """绘制ROC曲线"""
        plt.figure(figsize=figsize)

        # 处理二分类和多分类情况
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))

        if len(unique_labels) == 2:
            # 二分类ROC曲线
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(self.X_test)[:, 1]
            else:
                y_score = self.model.decision_function(self.X_test)

            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title('接收者操作特征曲线 (ROC)')
            plt.legend(loc="lower right")

        else:
            # 多分类ROC曲线
            # 二值化标签
            y_test_bin = label_binarize(self.y_test, classes=unique_labels)
            n_classes = len(unique_labels)

            # 获取预测概率
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(self.X_test)
            else:
                # 对于不支持predict_proba的模型，尝试使用decision_function
                try:
                    y_score = self.model.decision_function(self.X_test)
                    # 处理decision_function返回值维度不匹配的情况
                    if y_score.ndim == 1:
                        y_score = np.column_stack([1 - y_score, y_score])
                except:
                    raise ValueError("模型既不支持predict_proba也不支持decision_function")

            # 计算每个类的ROC曲线和AUC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title('多类别ROC曲线')
            plt.legend(loc="lower right")

        plt.show()

    def plot_learning_curve(self, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), figsize=(10, 6), scoring='accuracy'):
        """
        绘制学习曲线

        参数:
            cv: 交叉验证折数
            train_sizes: 训练集大小的比例
            figsize: 图形大小
            scoring: 评分方法
        """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=cv,
            train_sizes=train_sizes, scoring=scoring, n_jobs=-1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=figsize)
        plt.title('学习曲线')
        plt.xlabel('训练样本数')
        plt.ylabel(f'得分 ({scoring})')
        plt.grid()

        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集得分')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='验证集得分')

        plt.legend(loc='best')
        plt.show()

    def plot_calibration_curve(self, n_bins=10, figsize=(10, 8)):
        """
        绘制校准曲线

        参数:
            n_bins: 分箱数
            figsize: 图形大小
        """
        from sklearn.calibration import calibration_curve

        plt.figure(figsize=figsize)

        # 处理二分类和多分类
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))

        # 检查模型是否支持predict_proba
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("模型不支持predict_proba方法，无法绘制校准曲线")

        if len(unique_labels) == 2:
            # 二分类
            prob_pos = self.model.predict_proba(self.X_test)[:, 1]

            # 获取校准曲线
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, prob_pos, n_bins=n_bins)

            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label=f"{type(self.model).__name__}")

            plt.plot([0, 1], [0, 1], "k:", label="完美校准")

        else:
            # 多分类
            for i, class_name in enumerate(self.class_names):
                # 二值化标签
                y_test_bin = (self.y_test == i).astype(int)
                prob_pos = self.model.predict_proba(self.X_test)[:, i]

                # 获取校准曲线
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test_bin, prob_pos, n_bins=n_bins)

                plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                         label=f"{class_name}")

            plt.plot([0, 1], [0, 1], "k:", label="完美校准")

        plt.xlabel("预测概率")
        plt.ylabel("真实概率")
        plt.title("校准曲线")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(self, figsize=(10, 8)):
        """绘制精确率-召回率曲线"""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        plt.figure(figsize=figsize)

        # 处理二分类和多分类
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))

        if len(unique_labels) == 2:
            # 二分类PR曲线
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(self.X_test)[:, 1]
            else:
                y_score = self.model.decision_function(self.X_test)

            precision, recall, _ = precision_recall_curve(self.y_test, y_score)
            avg_precision = average_precision_score(self.y_test, y_score)

            plt.plot(recall, precision, lw=2,
                     label=f'精确率-召回率曲线 (AP = {avg_precision:.2f})')

        else:
            # 多分类PR曲线
            # 二值化标签
            y_test_bin = label_binarize(self.y_test, classes=unique_labels)
            n_classes = len(unique_labels)

            # 获取预测概率
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(self.X_test)
            else:
                # 对于不支持predict_proba的模型，尝试使用decision_function
                try:
                    y_score = self.model.decision_function(self.X_test)
                    # 处理decision_function返回值维度不匹配的情况
                    if y_score.ndim == 1:
                        y_score = np.column_stack([1 - y_score, y_score])
                except:
                    raise ValueError("模型既不支持predict_proba也不支持decision_function")

            # 计算每个类的PR曲线
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
                plt.plot(recall, precision, lw=2,
                         label=f'{self.class_names[i]} (AP = {avg_precision:.2f})')

        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('精确率-召回率曲线')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def generate_evaluation_report(self, output_file=None):
        """
        生成完整的评估报告

        参数:
            output_file: 输出文件路径 (可选)
        """
        metrics = self.calculate_metrics()

        # 创建报告内容
        report = f"""
        # 机器学习模型评估报告

        ## 模型信息
        - 模型类型: {type(self.model).__name__}

        ## 评估指标
        - 准确率 (Accuracy): {metrics['accuracy']:.4f}
        - 精确率 (Precision): {metrics['precision']:.4f}
        - 召回率 (Recall): {metrics['recall']:.4f}
        - F1分数 (F1-Score): {metrics['f1']:.4f}

        ## 分类报告
        ```
        {classification_report(self.y_test, self.y_pred, target_names=self.class_names)}
        ```
        """

        # 打印报告
        print(report)

        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到 {output_file}")

        return report

    def run_full_evaluation(self, output_file=None):
        """
        运行完整的评估流程并显示所有可视化

        参数:
            output_file: 评估报告输出文件路径 (可选)
        """
        # 1. 打印指标
        self.print_metrics()

        # 2. 绘制混淆矩阵
        print("\n绘制混淆矩阵...")
        self.plot_confusion_matrix()
        self.plot_confusion_matrix(normalize=True)

        # 3. 尝试绘制特征重要性
        print("\n绘制特征重要性...")
        try:
            self.plot_feature_importance(kind='model')
        except ValueError:
            try:
                self.plot_feature_importance(kind='permutation')
            except:
                print("无法绘制特征重要性，跳过...")

        # 4. 绘制特征分布
        print("\n绘制特征分布...")
        self.plot_feature_distributions()

        # 5. 绘制ROC曲线
        print("\n绘制ROC曲线...")
        try:
            self.plot_roc_curve()
        except:
            print("无法绘制ROC曲线，跳过...")

        # 6. 绘制精确率-召回率曲线
        print("\n绘制精确率-召回率曲线...")
        try:
            self.plot_precision_recall_curve()
        except:
            print("无法绘制精确率-召回率曲线，跳过...")

        # 7. 绘制学习曲线
        print("\n绘制学习曲线...")
        try:
            self.plot_learning_curve()
        except:
            print("无法绘制学习曲线，跳过...")

        # 8. 绘制校准曲线
        print("\n绘制校准曲线...")
        try:
            self.plot_calibration_curve()
        except:
            print("无法绘制校准曲线，跳过...")

        # 9. 生成评估报告
        if output_file:
            self.generate_evaluation_report(output_file)

        print("\n完整评估已完成!")


# 使用示例
def example_usage():
    """模块使用示例"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # 加载示例数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 初始化评估模块
    evaluator = MLEvaluationVisualization(
        model, X_train, X_test, y_train, y_test,
        class_names=class_names,
        feature_names=feature_names
    )

    # 运行完整评估
    evaluator.run_full_evaluation(output_file="model_evaluation_report.md")

    # 或者单独运行每个评估函数
    # metrics = evaluator.print_metrics()
    # evaluator.plot_confusion_matrix()
    # evaluator.plot_feature_importance()
    # evaluator.plot_feature_distributions()
    # evaluator.plot_roc_curve()
    # evaluator.plot_precision_recall_curve()
    # evaluator.plot_learning_curve()


if __name__ == "__main__":
    example_usage()