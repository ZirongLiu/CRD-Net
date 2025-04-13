import argparse
import csv
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from torch.nn.parallel import DistributedDataParallel
from dataset.dataset import get_multimodel_dataloaders
from utils.plot import plot_auc_curve, plot_confusion_matrix
from utils.matrix import specificity_score
from utils.utils import save_predictions, set_device
from model.model_list import create_model



def parse_args():
    parser = argparse.ArgumentParser(description="Training script for multiple models.")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes (default: 7)')
    parser.add_argument('--data_root', type=str, default='/data0/datasets/FEVRdataset/dataset', help='Root directory of dataset')
    parser.add_argument('--csv_file', type=str, default='/data0/datasets/FEVRdataset/dataset-train_test.csv', help='CSV file with train/test split')
    parser.add_argument('--modality', type=str, choices=['CF', 'FFA', 'MultiModel'], default='MultiModel', help='Modality to use (CF or FFA)')
    parser.add_argument('--output_dir', type=str, default='outputSGD', help='Directory to save models and logs')
    parser.add_argument('--gpus', type=str, default='1,3,5,7', help='Comma-separated list of GPU device IDs to use (e.g., "0,1,2")')
    return parser.parse_args()



def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir, model_name, modality):
    best_acc_50 = 0.0
    best_acc_100 = 0.0
    best_model_wts_50 = None
    best_model_wts_100 = None
    best_predictions_50 = None
    best_predictions_100 = None

    metrics_file = os.path.join(output_dir, modality, f"{model_name}_{modality}_metrics.csv")
    with open(metrics_file, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'accuracy', 'recall', 'precision', 'specificity', 'f1_score', 'kappa']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(num_epochs):
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs_CF, inputs_FFA, labels, path in train_progress:
                inputs_CF, inputs_FFA, labels = inputs_CF.to(device), inputs_FFA.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs_CF, inputs_FFA)
                if model_name == 'CRD-Net':
                    output_both, output_fundus, output_OCT = outputs
           
                    loss_both = criterion(output_both, labels)
         
                    loss_OCT = criterion(output_OCT, labels)
                    loss_fundus = criterion(output_fundus, labels)
   
                    loss = loss_both + loss_OCT + loss_fundus
                    pass
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs_CF.size(0)


                train_progress.set_postfix(loss=loss.item())

            train_loss = running_loss / len(train_loader.dataset)

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            all_logits = []
            sample_names = []
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
                for inputs_CF, inputs_FFA, labels, paths in val_progress:
                    inputs_CF, inputs_FFA, labels = inputs_CF.to(device), inputs_FFA.to(device), labels.to(device)
                    outputs = model(inputs_CF, inputs_FFA)

                    if model_name == 'CRD-Net':
                        output_both, output_fundus, output_OCT = outputs
         
                        loss_both = criterion(output_both, labels)

                        loss_OCT = criterion(output_OCT, labels)
                        loss_fundus = criterion(output_fundus, labels)

                        loss = loss_both + loss_OCT + loss_fundus
                        outputs = output_both
                    else:
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs_CF.size(0)
                    sample_names.extend(paths)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(outputs.softmax(dim=1).cpu().numpy())  
                    all_logits.extend(outputs.cpu().numpy())

                    val_progress.set_postfix(loss=loss.item())

            val_loss /= len(val_loader.dataset)
            val_accuracy = accuracy_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds, average='macro')
            precision = precision_score(all_labels, all_preds, average='macro')
            specificity = specificity_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            kappa = cohen_kappa_score(all_labels, all_preds)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Recall: {recall:.4f}, "
                  f"Precision: {precision:.4f}, Specificity: {specificity:.4f}, "
                  f"F1-Score: {f1:.4f}, Kappa: {kappa:.4f}")

            writer.writerow({
                'epoch': epoch + 1,
                'accuracy': val_accuracy,
                'recall': recall,
                'precision': precision,
                'specificity': specificity,
                'f1_score': f1,
                'kappa': kappa
            })

            # Save predictions and best models
            if epoch < 50 and kappa > best_acc_50:
                best_acc_50 = kappa
                best_model_wts_50 = model.state_dict()
                best_predictions_50 = (all_labels, all_preds, all_probs)
            if kappa > best_acc_100:
                best_acc_100 = kappa
                best_model_wts_100 = model.state_dict()
                best_predictions_100 = (all_labels, all_preds, all_probs)

        # Save the best weights
        if best_model_wts_50:
            torch.save(best_model_wts_50, os.path.join(output_dir, modality, f"{model_name}_{modality}_best_50.pth"))
            save_predictions(output_dir, modality, model_name+modality, 50, sample_names, all_labels, np.array(all_logits))
            plot_confusion_matrix(output_dir, modality, model_name+modality, 50, *best_predictions_50[:2], class_names=[str(i) for i in range(7)])
            plot_auc_curve(output_dir, modality, model_name+modality, 50, *best_predictions_50[::2])

        if best_model_wts_100:
            torch.save(best_model_wts_100, os.path.join(output_dir, modality, f"{model_name}_{modality}_best_100.pth"))
            save_predictions(output_dir, modality, model_name+modality, 100, sample_names, all_labels, np.array(all_logits))
            plot_confusion_matrix(output_dir, modality, model_name+modality, 100, *best_predictions_100[:2], class_names=[str(i) for i in range(7)])
            plot_auc_curve(output_dir, modality, model_name+modality, 100, *best_predictions_100[::2])


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, args.modality), exist_ok=True)
    device = set_device(args)  

    train_loader, val_loader = get_multimodel_dataloaders(args.data_root, args.csv_file, args.batch_size, test_type="all")

    models_to_train = {
        # 'CRD-Net': create_model('CRD-Net', args),
        # 'MSAN': create_model('MSAN', args),
        'MM-CNN': create_model('MM-CNN', args),
    }

    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")

        model = nn.DataParallel(model).to(device)

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

        train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.output_dir, model_name, args.modality)

if __name__ == "__main__":
    main()
