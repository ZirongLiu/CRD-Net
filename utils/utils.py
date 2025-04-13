import numpy as np
import csv
import os
import torch


def save_predictions(output_dir, modality, model_name, epoch, sample_names, y_true, y_logits):
    """
    保存预测结果到CSV。
    
    参数：
    - output_dir: 保存目录。
    - model_name: 模型名称。
    - epoch: 当前训练轮次。
    - sample_names: 样本名称或图片名称列表。
    - y_true: 真实标签。
    - y_logits: 模型输出的logits（未归一化得分）。
    """
    predictions_file = os.path.join(output_dir, modality, f"{model_name}_predictions_epoch_{epoch}.csv")

    # 计算softmax概率
    y_probs = np.exp(y_logits) / np.exp(y_logits).sum(axis=1, keepdims=True)

    # 获取预测类别
    y_pred = np.argmax(y_probs, axis=1)  # 直接从 softmax 后的概率取最大类别

    # 保存结果
    with open(predictions_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # CSV表头
        header = ['Sample_Name', 'True_Label', 'Predicted_Label'] + [f'Softmax_Class_{i}' for i in range(y_probs.shape[1])]
        writer.writerow(header)
        
        # 写入每一行数据
        for i, sample_name in enumerate(sample_names):
            row = [sample_name, y_true[i], y_pred[i]] + y_probs[i].tolist()
            writer.writerow(row)

    print(f"Saved predictions for epoch {epoch} to {predictions_file}")



def set_device(args):
    if args.gpus:
        # 设置指定的 GPU 设备
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using GPUs: {args.gpus}")
    else:
        # 默认使用所有可用的 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using all available GPUs or CPU.")
    return device