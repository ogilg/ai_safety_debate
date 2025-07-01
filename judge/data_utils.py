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

def get_sparse_image(revealed_pixels):
    """
    Create a sparse image representation from revealed pixels.
    Compatible with both the old and new representations.
    
    Args:
        revealed_pixels: List of (x, y, value) tuples
        
    Returns:
        Dictionary with keys 'indices' and 'values'
    """
    indices = []
    values = []
    
    for x, y, value in revealed_pixels:
        # Create a flat index (y * width + x)
        flat_idx = y * 28 + x
        indices.append(flat_idx)
        values.append(value)
        
    return {
        'indices': indices,
        'values': values
    }