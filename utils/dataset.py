import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

def get_cifar10(batch_size=64, subset_size=None):
    """
    Load the CIFAR-10 dataset with preprocessing and return DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        subset_size (int, optional): If set, restricts the training set to the
                                     first `subset_size` samples for faster experiments.

    Returns:
        (DataLoader, DataLoader): Tuple of (train_loader, test_loader).
    """
    # Normalize images to [-1, 1] and convert to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download/load training and test sets
    train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Optionally restrict training set size for debugging or prototyping
    if subset_size:
        train_set = Subset(train_set, list(range(subset_size)))

    # Create DataLoaders for iteration
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
