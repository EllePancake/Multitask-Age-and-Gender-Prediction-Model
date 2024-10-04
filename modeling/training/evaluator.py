import torch
from metrics import evaluate_model_on_loader
import pandas as pd


def evaluate_model(model, test_loader, device='cpu'):
    """
    Function to evaluate the model on the test dataset.

    Args:
        model (nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        metrics_dict (dict): Dictionary containing gender accuracy and age MAE.
    """
    model.to(device)
    model.eval()

    metrics = evaluate_model_on_loader(model, test_loader, device=device)

    print(f"Test Gender Accuracy: {metrics['gender_accuracy']:.4f}")
    print(f"Test Age MAE: {metrics['age_mae']:.4f}")

    return metrics


def load_model_from_checkpoint(model_class, checkpoint_path,
                               learning_rate=0.0001):
    """
    Load a model from a checkpoint.

    Args:
        model_class (nn.Module): The class of the model to be loaded.
        checkpoint_path (str): Path to the checkpoint file.
        learning_rate (float): Learning rate used to initialize the model.

    Returns:
        model (nn.Module): Loaded model with weights from the checkpoint.
    """
    model = model_class(learning_rate=learning_rate)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    checkpoint_state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key in checkpoint_state_dict.keys():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = checkpoint_state_dict[key]

    model.load_state_dict(new_state_dict, strict=False)

    return model


def evaluate_samples(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch

            images = images.to(device)

            if isinstance(targets, dict):
                age_targets = targets['age'].float().to(device)
                gender_targets = targets['gender'].long().to(device)
            else:
                age_targets = targets[:, 0].float().to(device)
                gender_targets = targets[:, 1].long().to(device)

            age_preds, gender_preds = model(images)
            age_preds_np = age_preds.cpu().numpy()
            gender_preds_np = torch.argmax(gender_preds, dim=1).cpu().numpy()

            for i in range(len(images)):
                gender_correct = (
                    gender_preds_np[i] == gender_targets[i].cpu().numpy()
                     )
                age_error = abs(age_preds_np[i] - age_targets[i].cpu().numpy())

                all_predictions.append((images[i], gender_correct, age_error))
                all_targets.append(targets)

    return all_predictions, all_targets


def get_best_and_worst_samples(all_predictions, num_samples=3):
    """
    Identify the best and worst performing samples.

    Args:
        all_predictions (list): List of tuples with (image, gender_correct,
        age_error).
        num_samples (int): Number of best and worst samples to return.

    Returns:
        list: Best performing samples.
        list: Worst performing samples.
    """
    sorted_predictions = sorted(all_predictions, key=lambda x: (x[1], -x[2]))

    best_samples = sorted_predictions[-num_samples:]
    worst_samples = sorted_predictions[:num_samples]

    return best_samples, worst_samples


if __name__ == "__main__":
    import argparse
    from base_model import MultiTaskResNet
    from utils_data_loader import (
        get_custom_data_loaders_v1,
        get_custom_data_loaders_v2,
        get_custom_data_loaders_v3,
        get_custom_data_loaders_v4
    )

    parser = argparse.ArgumentParser(
        description="Evaluate model version on test data.")
    parser.add_argument('--version', type=str, required=True,
                        help="Model version to evaluate (v1, v2, v3, v4)")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint.")

    args = parser.parse_args()
    version = args.version
    checkpoint_path = args.checkpoint_path

    if version == 'v1':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v1()
    elif version == 'v2':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v2()
    elif version == 'v3':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v3()
    elif version == 'v4':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v4()
    else:
        raise ValueError(f"Unsupported version: {version}")

    model = load_model_from_checkpoint(model_class, checkpoint_path,
                                       learning_rate=0.0001)

    metrics = evaluate_model(model, test_loader)

    metrics_dict = {
        'version': version,
        'test_gender_accuracy': metrics['gender_accuracy'],
        'test_age_mae': metrics['age_mae']
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv('results/evaluation_metrics.csv', mode='a',
                      header=False, index=False)

    print(f"Evaluation complete for {version}. Metrics saved.")
