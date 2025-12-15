"""
完整的RDGCNN可视化模块实现
提供训练过程可视化、结果分析和报告生成功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
import threading
import time


class RDGCNNVisualizer:
    """RDGCNN可视化器 - 提供训练和结果可视化功能"""

    def __init__(self, save_dir='./results', enable_real_time=False):
        """
        Args:
            save_dir: 保存目录
            enable_real_time: 是否启用实时可视化
        """
        self.save_dir = save_dir
        self.enable_real_time = enable_real_time
        self.real_time_enabled = False

        os.makedirs(save_dir, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")

        # 实时绘图相关
        if enable_real_time:
            self.setup_real_time_plotting()

        print(f"可视化器初始化完成，保存目录: {save_dir}")
        if enable_real_time:
            print("实时可视化已启用")

    def setup_real_time_plotting(self):
        """设置实时绘图"""
        try:
            plt.ion()  # 开启交互模式
            self.real_time_enabled = True
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('RDGCNN 训练实时监控', fontsize=16)
            plt.tight_layout()
        except Exception as e:
            print(f"实时绘图设置失败: {e}")
            self.real_time_enabled = False

    def update_real_time_plot(self, history):
        """更新实时训练图表"""
        if not self.real_time_enabled or not history.get('epochs'):
            return

        try:
            epochs = history['epochs']
            train_losses = history.get('train_losses', [])
            test_losses = history.get('test_losses', [])
            train_accs = history.get('train_accs', [])
            test_accs = history.get('test_accs', [])

            # 清除之前的图
            for ax in self.axes.flat:
                ax.clear()

            # 训练损失
            if train_losses and test_losses:
                self.axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
                self.axes[0, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
                self.axes[0, 0].set_title('Loss Curves')
                self.axes[0, 0].set_xlabel('Epoch')
                self.axes[0, 0].set_ylabel('Loss')
                self.axes[0, 0].legend()
                self.axes[0, 0].grid(True, alpha=0.3)

            # 训练准确率
            if train_accs and test_accs:
                self.axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
                self.axes[0, 1].plot(epochs, test_accs, 'r-', label='Test Acc', linewidth=2)
                self.axes[0, 1].set_title('Accuracy Curves')
                self.axes[0, 1].set_xlabel('Epoch')
                self.axes[0, 1].set_ylabel('Accuracy (%)')
                self.axes[0, 1].legend()
                self.axes[0, 1].grid(True, alpha=0.3)

            # 最佳准确率显示
            if test_accs:
                best_acc = max(test_accs)
                best_epoch = epochs[test_accs.index(best_acc)]
                self.axes[1, 0].text(0.5, 0.7, f'Best Test Accuracy',
                                   ha='center', va='center', fontsize=14, weight='bold')
                self.axes[1, 0].text(0.5, 0.5, f'{best_acc:.2f}%',
                                   ha='center', va='center', fontsize=24, weight='bold', color='red')
                self.axes[1, 0].text(0.5, 0.3, f'at Epoch {best_epoch}',
                                   ha='center', va='center', fontsize=12)
                self.axes[1, 0].set_xlim(0, 1)
                self.axes[1, 0].set_ylim(0, 1)
                self.axes[1, 0].axis('off')

            # 当前状态
            current_epoch = epochs[-1] if epochs else 0
            current_train_acc = train_accs[-1] if train_accs else 0
            current_test_acc = test_accs[-1] if test_accs else 0

            self.axes[1, 1].text(0.5, 0.8, f'Current Status',
                               ha='center', va='center', fontsize=14, weight='bold')
            self.axes[1, 1].text(0.5, 0.6, f'Epoch: {current_epoch}',
                               ha='center', va='center', fontsize=12)
            self.axes[1, 1].text(0.5, 0.4, f'Train: {current_train_acc:.2f}%',
                               ha='center', va='center', fontsize=12)
            self.axes[1, 1].text(0.5, 0.2, f'Test: {current_test_acc:.2f}%',
                               ha='center', va='center', fontsize=12)
            self.axes[1, 1].set_xlim(0, 1)
            self.axes[1, 1].set_ylim(0, 1)
            self.axes[1, 1].axis('off')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        except Exception as e:
            print(f"实时绘图更新失败: {e}")

    def disable_real_time_plotting(self):
        """关闭实时绘图"""
        if self.real_time_enabled:
            plt.ioff()
            plt.close(self.fig)
            self.real_time_enabled = False
            print("实时可视化已关闭")

    def plot_training_history(self, history, save_path=None):
        """绘制训练历史曲线"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_history.png')

        epochs = history.get('epochs', [])
        if not epochs:
            print("警告: 没有训练历史数据")
            return

        train_losses = history.get('train_losses', [])
        test_losses = history.get('test_losses', [])
        train_accs = history.get('train_accs', [])
        test_accs = history.get('test_accs', [])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 损失曲线
        if train_losses and test_losses:
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Test Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 准确率曲线
        if train_accs and test_accs:
            ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
            ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Test Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 标记最佳点
            if test_accs:
                best_acc = max(test_accs)
                best_epoch = epochs[test_accs.index(best_acc)]
                ax2.plot(best_epoch, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.2f}%')
                ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练历史曲线已保存到: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """绘制混淆矩阵"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建图像
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm_normalized,
                    annot=False,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Normalized Frequency'})

        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title('Confusion Matrix - RDGCNN on ModelNet40', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存到: {save_path}")

    def plot_class_accuracies(self, class_accuracies, class_names, save_path=None):
        """绘制各类别准确率"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'class_accuracies.png')

        # 处理输入数据
        if isinstance(class_accuracies, dict):
            # 如果是字典，转换为列表
            accs = [class_accuracies.get(i, 0.0) for i in range(len(class_names))]
        else:
            accs = list(class_accuracies)

        # 创建图像
        plt.figure(figsize=(20, 8))

        x = np.arange(len(class_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

        bars = plt.bar(x, accs, color=colors, alpha=0.8, edgecolor='black')

        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Per-Class Accuracy - RDGCNN on ModelNet40', fontsize=16)
        plt.xticks(x, class_names, rotation=90, fontsize=10)
        plt.ylim([0, 1])

        # 添加平均准确率线
        mean_acc = np.mean(accs)
        plt.axhline(y=mean_acc, color='r', linestyle='--',
                   linewidth=2, label=f'Mean Accuracy: {mean_acc:.4f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"类别准确率图已保存到: {save_path}")

    def create_training_report(self, history, model_info=None, config=None, save_path=None):
        """创建训练报告"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_report.json')

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = {
            "timestamp": timestamp,
            "training_summary": {
                "total_epochs": len(history.get('epochs', [])),
                "best_test_accuracy": history.get('best_acc', 0.0),
                "best_epoch": history.get('best_epoch', 0),
                "final_train_accuracy": history.get('train_accs', [0])[-1] if history.get('train_accs') else 0,
                "final_test_accuracy": history.get('test_accs', [0])[-1] if history.get('test_accs') else 0,
                "final_train_loss": history.get('train_losses', [0])[-1] if history.get('train_losses') else 0,
                "final_test_loss": history.get('test_losses', [0])[-1] if history.get('test_losses') else 0
            },
            "model_info": model_info or {},
            "config": config or {},
            "history": history
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"训练报告已保存到: {save_path}")
        return report

    def plot_parameter_analysis(self, model, save_path=None):
        """绘制参数分析图"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'parameter_analysis.png')

        if not hasattr(model, 'get_parameter_info'):
            print("模型不支持参数分析")
            return

        param_info = model.get_parameter_info()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 参数数量饼图
        labels = ['Trainable', 'Frozen']
        sizes = [param_info['trainable_params'], param_info['frozen_params']]
        colors = ['#ff9999', '#66b3ff']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Parameter Distribution')

        # 参数数量柱状图
        categories = ['Total', 'Trainable', 'Frozen']
        values = [param_info['total_params'], param_info['trainable_params'], param_info['frozen_params']]

        ax2.bar(categories, values, color=['green', 'blue', 'red'], alpha=0.7)
        ax2.set_title('Parameter Counts')
        ax2.set_ylabel('Number of Parameters')

        # 添加数值标签
        for i, v in enumerate(values):
            ax2.text(i, v + max(values) * 0.01, f'{v:,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"参数分析图已保存到: {save_path}")


def create_visualizer(save_dir='./results', enable_real_time=False):
    """创建可视化器的便捷函数"""
    return RDGCNNVisualizer(save_dir=save_dir, enable_real_time=enable_real_time)


# 示例使用
if __name__ == '__main__':
    # 创建示例可视化器
    visualizer = create_visualizer('./test_results', enable_real_time=True)

    # 示例训练历史数据
    dummy_history = {
        'epochs': list(range(1, 21)),
        'train_losses': [0.8 - i*0.03 for i in range(20)],
        'test_losses': [0.9 - i*0.025 for i in range(20)],
        'train_accs': [70 + i*1.5 for i in range(20)],
        'test_accs': [65 + i*1.2 for i in range(20)],
        'best_acc': 88.5,
        'best_epoch': 18
    }

    # 生成可视化
    visualizer.plot_training_history(dummy_history)

    # 示例混淆矩阵数据
    np.random.seed(42)
    n_classes = 10
    n_samples = 1000
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    class_names = [f'Class_{i}' for i in range(n_classes)]

    visualizer.plot_confusion_matrix(y_true, y_pred, class_names)

    # 示例类别准确率
    class_accs = np.random.uniform(0.6, 0.95, n_classes)
    visualizer.plot_class_accuracies(class_accs, class_names)

    # 创建训练报告
    visualizer.create_training_report(dummy_history)

    print("可视化模块测试完成!")