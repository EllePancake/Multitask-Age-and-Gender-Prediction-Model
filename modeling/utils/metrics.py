import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error


def calculate_gender_accuracy(gender_preds, gender_targets):
    """
    Calculate the accuracy for gender classification.

    Args:
        gender_preds (torch.Tensor): Predicted gender values (logits).
        gender_targets (torch.Tensor): True gender labels.

    Returns:
        accuracy (float): Accuracy of the gender classification task.
    """
    predicted_classes = torch.argmax(gender_preds, dim=1)
    predicted_classes_np = predicted_classes.cpu().numpy()
    gender_targets_np = gender_targets.cpu().numpy()

    accuracy = accuracy_score(gender_targets_np, predicted_classes_np)
    return accuracy


def calculate_age_mae(age_preds, age_targets):
    """
    Calculate the Mean Absolute Error (MAE) for age regression.

    Args:
        age_preds (torch.Tensor): Predicted age values.
        age_targets (torch.Tensor): True age labels.

    Returns:
        mae (float): Mean Absolute Error of the age regression task.
    """
    age_preds_np = age_preds.squeeze().cpu().detach().numpy()
    age_targets_np = age_targets.cpu().detach().numpy()

    mae = mean_absolute_error(age_targets_np, age_preds_np)
    return mae


def log_metrics(trainer, gender_preds, gender_targets, age_preds,
                age_targets, stage='train'):
    """
    Log metrics for gender classification and age regression.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer object to log
        metrics.
        gender_preds (torch.Tensor): Predicted gender values (logits).
        gender_targets (torch.Tensor): True gender labels.
        age_preds (torch.Tensor): Predicted age values.
        age_targets (torch.Tensor): True age labels.
        stage (str): The stage of the model ('train', 'val', 'test').
    """
    gender_accuracy = calculate_gender_accuracy(gender_preds, gender_targets)
    age_mae = calculate_age_mae(age_preds, age_targets)

    trainer.log(f'{stage}_gender_accuracy', gender_accuracy, prog_bar=True)
    trainer.log(f'{stage}_age_mae', age_mae, prog_bar=True)

    print(f'{stage.capitalize()} Gender Accuracy: {gender_accuracy:.4f}')
    print(f'{stage.capitalize()} Age MAE: {age_mae:.4f}')


def evaluate_model_on_loader(model, data_loader, device='cpu'):
    """
    Evaluate the model on a given data loader.

    Args:
        model (nn.Module): The trained PyTorch model to be evaluated.
        data_loader (DataLoader): The DataLoader object containing the dataset.
        device (str): The device on which to perform evaluation ('cpu' or
        'cuda').

    Returns:
        dict: A dictionary containing gender accuracy and age MAE.
    """
    model.eval()
    gender_preds_all = []
    gender_targets_all = []
    age_preds_all = []
    age_targets_all = []

    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch
            images = images.to(device)

            if isinstance(targets, dict):
                age_targets = targets['age'].float().to(device)
                gender_targets = targets['gender'].long().to(device)
            elif isinstance(targets, (list, tuple)) and len(targets) == 2:
                age_targets = targets[0].float().to(device)
                gender_targets = targets[1].long().to(device)
            elif isinstance(targets, torch.Tensor):
                try:
                    age_targets = targets[:, 0].float().to(device)
                    gender_targets = targets[:, 1].long().to(device)
                except Exception as e:
                    print(f"Error accessing tensor targets: {e}")
                    raise
            else:
                raise TypeError(f"Unexpected targets format: {type(targets)}")

            age_preds, gender_preds = model(images)

            gender_preds_all.append(gender_preds)
            gender_targets_all.append(gender_targets)
            age_preds_all.append(age_preds)
            age_targets_all.append(age_targets)

    gender_preds_all = torch.cat(gender_preds_all, dim=0)
    gender_targets_all = torch.cat(gender_targets_all, dim=0)
    age_preds_all = torch.cat(age_preds_all, dim=0)
    age_targets_all = torch.cat(age_targets_all, dim=0)

    gender_accuracy = calculate_gender_accuracy(gender_preds_all,
                                                gender_targets_all)
    age_mae = calculate_age_mae(age_preds_all, age_targets_all)

    return {
        'gender_accuracy': gender_accuracy,
        'age_mae': age_mae
    }
