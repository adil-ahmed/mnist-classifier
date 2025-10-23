import torch
from torchvision import datasets, transforms

# MNIST mean and std for normalization
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def get_loaders(batch_size=64, augment=False):
    # Basic transformations 
    # PyTorch converts them to values in [0, 1]
    tfms_train = [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    if augment:
        # Apply slight random rotation or shift
        # Prevents overfitting and teaches the model to recognize digits even if they’re slightly moved, tilted, or written differently.
        tfms_train = [transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))] + tfms_train

    tfms_test = [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]

    # Load datasets
    train_set = datasets.MNIST(root="./data", train=True, transform=transforms.Compose(tfms_train), download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transforms.Compose(tfms_test), download=True)

    # Wrap in DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

