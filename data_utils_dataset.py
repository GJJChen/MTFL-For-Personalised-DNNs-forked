import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gzip
import os


class CustomMNISTDataset(Dataset):
    """
    Custom dataset for MNIST that supports adding noise and non-IID data splitting.
    """

    def __init__(self, data, labels, transform=None):
        """
        Initialize the dataset.

        Args:
            data (np.ndarray): The images.
            labels (np.ndarray): The labels of the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_mnist(data_dir, transform=None):
    """
    Load MNIST dataset from the specified directory.

    Args:
        data_dir (str): Directory where MNIST files are stored.
        transform (callable, optional): Transformations to apply to the data.

    Returns:
        CustomMNISTDataset: Dataset object for MNIST.
    """

    def read_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip the header
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    def read_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip the header
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = read_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = read_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = read_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = read_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # Apply default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = CustomMNISTDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomMNISTDataset(test_images, test_labels, transform=transform)

    return train_dataset, test_dataset


def get_data_loaders(train_dataset, test_dataset, batch_size=64, num_workers=4):
    """
    Get data loaders for the training and test datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The test dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: A tuple containing the training DataLoader and test DataLoader.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
