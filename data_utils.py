import torch
import numpy as np
from torchvision import datasets, transforms

def get_mnist_test_dataset(data_dir='./data'):
    """
    Load the MNIST test dataset with proper transformations.
    
    Args:
        data_dir: Directory to store/load the dataset
        
    Returns:
        test_dataset: The MNIST test dataset with transformations applied
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return test_dataset

def get_mnist_train_dataset(data_dir='./data'):
    """
    Load the MNIST training dataset with proper transformations.
    
    Args:
        data_dir: Directory to store/load the dataset
        
    Returns:
        train_dataset: The MNIST training dataset with transformations applied
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return train_dataset 

def get_sparse_image(revealed_pixels, image_size=(28, 28)):
    """Create sparse image with only revealed pixels using vectorized operations"""
    sparse_image = np.zeros(image_size)
    
    # Extract coordinates and values from revealed_pixels
    if revealed_pixels:
        x_coords, y_coords, values = zip(*revealed_pixels)
        sparse_image[y_coords, x_coords] = values
        
    return sparse_image