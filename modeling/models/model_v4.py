from base_model import MultiTaskResNet
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch


def get_data_loaders_v4(batch_size=32):
    """
    Function to create DataLoader objects for train, validation, and test
    datasets for version 4.
    Adds an imbalanced sampler to handle class imbalance.

    Args:
        batch_size (int): The size of the batch to be used during training/
        testing.

    Returns:
        train_loader (DataLoader): DataLoader object for training set with
        imbalanced sampler.
        val_loader (DataLoader): DataLoader object for validation set.
        test_loader (DataLoader): DataLoader object for test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(root='data/train', transform=transform)
    val_dataset = ImageFolder(root='data/validation', transform=transform)
    test_dataset = ImageFolder(root='data/test', transform=transform)

    targets = [sample[1] for sample in train_dataset.samples]
    class_counts = torch.tensor([targets.count(i) for i in range(
        len(train_dataset.classes))], dtype=torch.float)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[target] for target in targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    learning_rate = 0.0001
    batch_size = 32

    model = MultiTaskResNet(learning_rate=learning_rate)

    train_loader, val_loader, test_loader = get_data_loaders_v4(
        batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint("models/checkpoints/model_v4.ckpt")
