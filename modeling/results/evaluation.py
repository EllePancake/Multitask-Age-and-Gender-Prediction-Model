import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score


def generate_classification_report(model, loader, dataset_name="Test",
                                   device='cpu'):
    all_gender_targets = []
    all_gender_preds = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, targets = batch
            gender_targets = targets['gender']
            images = images.to(device)
            gender_targets = gender_targets.to(device)

            _, gender_preds = model(images)
            gender_preds = torch.argmax(gender_preds, dim=1)

            all_gender_targets.extend(gender_targets.cpu().numpy())
            all_gender_preds.extend(gender_preds.cpu().numpy())

    report = classification_report(all_gender_targets, all_gender_preds,
                                   target_names=['Male', 'Female'])
    print(f"Classification Report for {dataset_name} Set:")
    print(report)


def perform_threshold_analysis(model, loader, device='cpu'):
    all_gender_targets = []
    all_gender_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, targets = batch
            gender_targets = targets['gender']
            images = images.to(device)
            gender_targets = gender_targets.to(device)

            _, gender_preds = model(images)
            gender_probs = torch.softmax(gender_preds, dim=1)[:, 1]

            all_gender_targets.extend(gender_targets.cpu().numpy())
            all_gender_probs.extend(gender_probs.cpu().numpy())

    all_gender_targets = np.array(all_gender_targets)
    all_gender_probs = np.array(all_gender_probs)

    precision, recall, thresholds = precision_recall_curve(all_gender_targets,
                                                           all_gender_probs)
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve Analysis')
    plt.legend()
    plt.grid()
    plt.show()

    fpr, tpr, roc_thresholds = roc_curve(all_gender_targets, all_gender_probs)
    roc_auc = roc_auc_score(all_gender_targets, all_gender_probs)
    plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.show()

    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    print(f'Best Threshold (based on F1 score): {best_threshold}')


def plot_confusion_matrix(model, loader, dataset_name="Test", device='cpu'):
    all_gender_targets = []
    all_gender_preds = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, targets = batch
            gender_targets = targets['gender']
            images = images.to(device)
            gender_targets = gender_targets.to(device)

            _, gender_preds = model(images)
            gender_preds = torch.argmax(gender_preds, dim=1)

            all_gender_targets.extend(gender_targets.cpu().numpy())
            all_gender_preds.extend(gender_preds.cpu().numpy())

    cm = confusion_matrix(all_gender_targets, all_gender_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Male', 'Female'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {dataset_name} Set')
    plt.show()
