import pandas as pd
import matplotlib.pyplot as plt


def plot_training_validation_loss(metrics_file='results/metrics.csv'):
    """
    Plot training and validation loss for each model version.

    Args:
        metrics_file (str): Path to the CSV file containing training and
        validation metrics.
    """
    if not metrics_file or not pd.io.common.file_exists(metrics_file):
        print(f"Metrics file '{metrics_file}' does not exist.")
        return

    metrics_df = pd.read_csv(metrics_file, header=None)
    metrics_df.columns = ['Version', 'Avg_Train_Loss', 'Avg_Val_Loss',
                          'Train_Gender_Accuracy', 'Train_Age_MAE',
                          'Val_Gender_Accuracy', 'Val_Age_MAE']

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['Version'], metrics_df['Avg_Train_Loss'],
             label='Avg Train Loss', marker='o', linestyle='-')
    plt.plot(metrics_df['Version'], metrics_df['Avg_Val_Loss'],
             label='Avg Val Loss', marker='o', linestyle='-')
    plt.xlabel('Model Version')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_metrics(metrics_file='results/metrics.csv'):
    """
    Plot validation metrics (gender accuracy and age MAE) for each model
    version.

    Args:
        metrics_file (str): Path to the CSV file containing training and
        validation metrics.
    """
    # Load metrics from CSV file
    if not metrics_file or not pd.io.common.file_exists(metrics_file):
        print(f"Metrics file '{metrics_file}' does not exist.")
        return

    metrics_df = pd.read_csv(metrics_file, header=None)
    metrics_df.columns = ['Version', 'Avg_Train_Loss', 'Avg_Val_Loss',
                          'Train_Gender_Accuracy', 'Train_Age_MAE',
                          'Val_Gender_Accuracy', 'Val_Age_MAE']

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['Version'], metrics_df['Val_Gender_Accuracy'],
             label='Validation Gender Accuracy', marker='o', linestyle='-')
    plt.plot(metrics_df['Version'], metrics_df['Val_Age_MAE'],
             label='Validation Age MAE', marker='o', linestyle='-')
    plt.xlabel('Model Version')
    plt.ylabel('Metrics')
    plt.title('Validation Metrics Comparison (Gender Accuracy and Age MAE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_test_metrics(evaluation_file='results/evaluation_metrics.csv'):
    """
    Plot test metrics (gender accuracy and age MAE) for each model version.

    Args:
        evaluation_file (str): Path to the CSV file containing test evaluation
        metrics.
    """
    if not evaluation_file or not pd.io.common.file_exists(evaluation_file):
        print(f"Evaluation metrics file '{evaluation_file}' does not exist.")
        return

    evaluation_df = pd.read_csv(evaluation_file, header=None)
    evaluation_df.columns = ['Version', 'Test_Gender_Accuracy', 'Test_Age_MAE']

    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_df['Version'], evaluation_df['Test_Gender_Accuracy'],
             label='Test Gender Accuracy', marker='o', linestyle='-')
    plt.plot(evaluation_df['Version'], evaluation_df['Test_Age_MAE'],
             label='Test Age MAE', marker='o', linestyle='-')
    plt.xlabel('Model Version')
    plt.ylabel('Metrics')
    plt.title('Test Metrics Comparison (Gender Accuracy and Age MAE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_over_versions():
    """
    Helper function to call all the plotting functions to visualize metrics at
    once.
    """
    print("Plotting Training and Validation Loss...")
    plot_training_validation_loss()

    print("\nPlotting Validation Metrics...")
    plot_validation_metrics()

    print("\nPlotting Test Metrics...")
    plot_test_metrics()


if __name__ == "__main__":
    plot_metrics_over_versions()
