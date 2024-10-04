import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torchvision.transforms as transforms
from PIL import Image


class CustomUTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset for UTKFace images.

        Args:
            root_dir (str): Directory with all the images (organized in gender
        subfolders).
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_files.append(os.path.join(subdir, file))

        print(f"[DEBUG] Loaded {len(self.image_files)} images from {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        filename = os.path.basename(img_path)
        filename_parts = filename.split('_')
        age = int(filename_parts[0])
        gender = int(filename_parts[1])
        ethnicity = int(filename_parts[2])

        labels = {
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity
        }

        if self.transform:
            image = self.transform(image)

        return image, labels


def get_custom_data_loaders(batch_size=32, version='v1'):
    """
    Core function to create DataLoader objects for train, validation, and test
    datasets based on the specified version.

    Args:
        batch_size (int): Size of the batch to be used during training/testing.
        version (str): The version of the model, which determines the data
        transformation/sampler strategy.

    Returns:
        train_loader (DataLoader): DataLoader object for training set.
        val_loader (DataLoader): DataLoader object for validation set.
        test_loader (DataLoader): DataLoader object for test set.
    """
    transform_list = [transforms.ToTensor()]

    if version in ['v2', 'v3', 'v4', 'v5']:
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]))

    transform = transforms.Compose(transform_list)

    train_dataset = CustomUTKFaceDataset(root_dir='data/train',
                                         transform=transform)
    val_dataset = CustomUTKFaceDataset(root_dir='data/validation',
                                       transform=transform)
    test_dataset = CustomUTKFaceDataset(root_dir='data/test',
                                        transform=transform)

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    if version == 'v4' and len(train_dataset) > 0:
        targets = [sample['gender'] for _, sample in train_dataset]
        class_counts = torch.tensor(
            [targets.count(i) for i in range(2)],
            dtype=torch.float
        )

        class_weights = 1. / class_counts
        sample_weights = [class_weights[target] for target in targets]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader


def get_custom_data_loaders_v1(batch_size=32):
    """
    Wrapper for getting DataLoader objects for version 1 (no normalization).
    """
    return get_custom_data_loaders(batch_size=batch_size, version='v1')


def get_custom_data_loaders_v2(batch_size=32):
    """
    Wrapper for getting DataLoader objects for version 2 (with normalization).
    """
    return get_custom_data_loaders(batch_size=batch_size, version='v2')


def get_custom_data_loaders_v3(batch_size=32):
    """
    Wrapper for getting DataLoader objects for version 3 (with normalization
    and reduced learning rate).
    """
    return get_custom_data_loaders(batch_size=batch_size, version='v3')


def get_custom_data_loaders_v4(batch_size=32):
    """
    Wrapper for getting DataLoader objects for version 4 (with normalization
    and imbalanced sampler).
    """
    return get_custom_data_loaders(batch_size=batch_size, version='v4')
