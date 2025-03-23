import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import get_mnist_train_dataset, get_mnist_test_dataset

# Define the MLP model architecture
class MNISTJudge(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, dropout_rate=0.2, model_type='mlp'):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == 'mlp':
            layers = []
            prev_size = input_size
            
            # Build hidden layers dynamically based on hidden_sizes
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, output_size))
            
            self.model = nn.Sequential(*layers)
        
        elif model_type == 'cnn':
            # CNN architecture
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate)
            )
            
            # Calculate the size after convolutions and pooling
            # Input: 28x28 -> Conv -> 28x28 -> MaxPool -> 14x14
            # -> Conv -> 14x14 -> MaxPool -> 7x7 with 64 channels
            conv_output_size = 7 * 7 * 64
            
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_sizes[0], output_size)
            )
    
    def forward(self, x):
        if self.model_type == 'mlp':
            return self.model(x)
        elif self.model_type == 'cnn':
            # Reshape input to [batch_size, channels, height, width]
            if x.dim() == 2:
                x = x.view(-1, 1, 28, 28)
            
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc_layers(x)
            return x

# Custom dataset for sparse MNIST with fixed number of pixels
class SparseMNIST(Dataset):
    def __init__(self, mnist_dataset, num_pixels=10, model_type='mlp', sampling_mode='random'):
        """
        Args:
            mnist_dataset: The original MNIST dataset
            num_pixels: Number of pixels to reveal
            model_type: 'mlp' or 'cnn'
            sampling_mode: How to sample pixels to reveal
                - 'random': Completely random pixels
                - 'nonzero': Random non-zero pixels only
                - 'weighted': Pixels sampled according to their frequency across dataset
        """
        self.mnist_dataset = mnist_dataset
        self.num_pixels = num_pixels
        self.model_type = model_type
        self.sampling_mode = sampling_mode
        
        # For weighted sampling, compute pixel frequency distribution across dataset
        if sampling_mode == 'weighted':
            self.pixel_weights = self._compute_pixel_weights()
    
    def _compute_pixel_weights(self):
        """Compute the frequency of each pixel being non-zero across the dataset"""
        print("Computing pixel frequency distribution across dataset...")
        pixel_counts = torch.zeros(784)
        
        # Sample a subset of images to compute weights (for efficiency)
        num_samples = min(10000, len(self.mnist_dataset))
        indices = np.random.choice(len(self.mnist_dataset), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="Computing pixel weights"):
            image, _ = self.mnist_dataset[idx]
            flat_image = image.view(-1)
            # Count non-zero pixels
            pixel_counts += (flat_image > 0).float()
        
        # Normalize to get probability distribution
        pixel_weights = pixel_counts / num_samples
        # Add small epsilon to avoid zero probabilities
        pixel_weights = pixel_weights + 1e-5
        pixel_weights = pixel_weights / pixel_weights.sum()
        
        return pixel_weights
    
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        
        # Convert image to flat tensor if it's not already
        if len(image.shape) > 1:
            flat_image = image.view(-1)
        else:
            flat_image = image
            
        # Create a sparse version by keeping only a fixed number of pixels
        num_pixels = flat_image.shape[0]
        
        # Create a mask of zeros
        sparse_mask = torch.zeros_like(flat_image)
        
        # Select visible pixels based on sampling mode
        if self.sampling_mode == 'random':
            # Completely random pixels
            visible_indices = np.random.choice(num_pixels, self.num_pixels, replace=False)
            
        elif self.sampling_mode == 'nonzero':
            # Random non-zero pixels only
            nonzero_indices = torch.nonzero(flat_image > 0).squeeze().cpu().numpy()
            
            # If there are fewer non-zero pixels than requested, use all of them
            if len(nonzero_indices) <= self.num_pixels:
                visible_indices = nonzero_indices
            else:
                visible_indices = np.random.choice(nonzero_indices, self.num_pixels, replace=False)
                
        elif self.sampling_mode == 'weighted':
            # Sample according to the pre-computed distribution
            visible_indices = np.random.choice(
                num_pixels, 
                self.num_pixels, 
                replace=False, 
                p=self.pixel_weights.numpy()
            )
        
        # Set the selected pixels to 1 in the mask
        sparse_mask[visible_indices] = 1
        
        # Apply mask to get sparse image
        sparse_image = flat_image * sparse_mask
        
        # For CNN, we need to keep the 2D structure
        if self.model_type == 'cnn':
            sparse_image = sparse_image.view(1, 28, 28)
        
        return sparse_image, label

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train MNIST Judge Model')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[128, 64], help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num-pixels', type=int, default=10, help='Fixed number of visible pixels')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'cnn'], 
                        help='Model architecture: mlp or cnn')
    parser.add_argument('--sampling-mode', type=str, default='random', 
                        choices=['random', 'nonzero', 'weighted'],
                        help='How to sample pixels: random, nonzero, or weighted')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST datasets using data_utils
    train_dataset = get_mnist_train_dataset('./data')
    test_dataset = get_mnist_test_dataset('./data')
    
    # Create sparse versions with fixed number of pixels
    sparse_train_dataset = SparseMNIST(
        train_dataset, 
        num_pixels=args.num_pixels, 
        model_type=args.model_type,
        sampling_mode=args.sampling_mode
    )
    
    # For evaluation, we use the same sampling mode
    sparse_test_dataset = SparseMNIST(
        test_dataset, 
        num_pixels=args.num_pixels, 
        model_type=args.model_type,
        sampling_mode=args.sampling_mode
    )
    
    # Create data loaders
    train_loader = DataLoader(sparse_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(sparse_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = MNISTJudge(
        input_size=784,  # 28x28 pixels
        hidden_sizes=args.hidden_sizes,
        output_size=10,  # 10 digits
        dropout_rate=args.dropout,
        model_type=args.model_type
    ).to(device)
    
    # Create a model name that includes the sampling mode
    model_name = f"{args.model_type}_{args.sampling_mode}_{args.num_pixels}px"
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'best_model_{model_name}.pth'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Plot and save training metrics
    plot_metrics(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(args.save_dir, f'training_metrics_{model_name}.png')
    )
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_accs[-1],
        'train_acc': train_accs[-1],
        'val_loss': val_losses[-1],
        'train_loss': train_losses[-1],
        'args': vars(args)
    }, os.path.join(args.save_dir, f'final_model_{model_name}.pth'))
    
    print(f"Training completed. Final validation accuracy: {val_accs[-1]:.2f}%")

if __name__ == '__main__':
    main() 