import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import os
import numpy as np



def plot_confusion_matrix(output_dir, modality, model_name, epoch, y_true, y_pred, class_names):
    # 绘制混淆矩阵图
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.savefig(os.path.join(output_dir, modality, f"{model_name}_confusion_matrix_epoch_{epoch}.png"))
    plt.close()
    print(f"Saved confusion matrix for epoch {epoch}")

def plot_auc_curve(output_dir, modality, model_name, epoch, y_true, y_score):
    """
    绘制多分类的AUC曲线。
    y_true: 真实的分类标签（0, 1, 2, ..., num_classes-1）
    y_score: 模型的输出概率矩阵，每列为每个类别的概率
    """
    # 确保 y_score 是 NumPy 数组
    y_score = np.array(y_score)
    print(f"y_score shape: {y_score.shape}")

    # 将 y_true 转换为 NumPy 数组，确保支持 .astype(int)
    y_true = np.array(y_true)
    print(f"y_true type: {type(y_true)}, content: {y_true}")

    num_classes = y_score.shape[1]
    class_names = [str(i) for i in range(num_classes)]

    # 创建绘图
    plt.figure()
    
    for i in range(num_classes):
        # 将当前类别视为正类，其余类别视为负类
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)

        # 绘制每个类别的ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'{i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Multiclass ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(output_dir, modality, f"{model_name}_roc_curve_epoch_{epoch}.png"))
    plt.close()
    print(f"Saved ROC curve for epoch {epoch}")




