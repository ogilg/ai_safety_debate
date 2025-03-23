import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from train_judge import MNISTJudge, SparseMNIST
from data_utils import get_mnist_test_dataset

def visualize_predictions(model, test_dataset, device, num_samples=5, save_path=None):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(test_dataset))
        sparse_image, true_label = test_dataset[idx]
        
        # Reshape for visualization
        original_shape = (28, 28)
        sparse_image_2d = sparse_image.view(*original_shape).cpu().numpy()
        
        # Make prediction
        with torch.no_grad():
            input_tensor = sparse_image.unsqueeze(0).to(device)
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = output.argmax(dim=1).item()
        
        # Plot sparse image
        axes[i, 0].imshow(sparse_image_2d, cmap='gray')
        axes[i, 0].set_title(f'Sparse Image (True: {true_label})')
        axes[i, 0].axis('off')
        
        # Plot prediction probabilities
        probs = probabilities[0].cpu().numpy()
        axes[i, 1].bar(range(10), probs)
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title(f'Prediction: {predicted_label} (Confidence: {probs[predicted_label]:.2f})')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_by_sparsity(model, test_dataset, device, sparsity_levels=None, sampling_mode='random'):
    if sparsity_levels is None:
        sparsity_levels = [5, 10]
    
    results = {}
    model.eval()
    
    # Get model type from the model
    model_type = model.model_type if hasattr(model, 'model_type') else 'mlp'
    
    for num_pixels in sparsity_levels:
        print(f"Evaluating with {num_pixels} visible pixels using {sampling_mode} sampling...")
        
        # Create dataset with fixed number of pixels and specified sampling mode
        fixed_sparsity_dataset = SparseMNIST(
            test_dataset.mnist_dataset, 
            num_pixels=num_pixels,
            model_type=model_type,
            sampling_mode=sampling_mode
        )
        test_loader = DataLoader(fixed_sparsity_dataset, batch_size=128, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        results[num_pixels] = accuracy
        print(f"Accuracy with {num_pixels} pixels: {accuracy:.2f}%")
    
    return results

def plot_sparsity_results(results, save_path=None):
    pixels = sorted(list(results.keys()))
    accuracies = [results[p] for p in pixels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(pixels, accuracies, 'o-')
    plt.xlabel('Number of Visible Pixels')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs. Number of Visible Pixels')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST Judge Model')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num-pixels', type=int, default=10, help='Fixed number of visible pixels')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--sparsity-levels', type=int, nargs='+', default=[5, 10, 15, 20, 30, 40, 50], 
                        help='Sparsity levels to evaluate')
    parser.add_argument('--sampling-mode', type=str, default=None, 
                        choices=['random', 'nonzero', 'weighted', None],
                        help='Override the sampling mode (default: use the one from the model)')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # Get model type and sampling mode (default to 'mlp' and 'random' for backward compatibility)
    model_type = model_args.get('model_type', 'mlp')
    sampling_mode = args.sampling_mode or model_args.get('sampling_mode', 'random')
    
    model = MNISTJudge(
        input_size=784,
        hidden_sizes=model_args['hidden_sizes'],
        output_size=10,
        dropout_rate=model_args['dropout'],
        model_type=model_type
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    print(f"Model type: {model_type}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Load MNIST test dataset using data_utils
    test_dataset = get_mnist_test_dataset('./data')
    sparse_test_dataset = SparseMNIST(
        test_dataset, 
        num_pixels=args.num_pixels, 
        model_type=model_type,
        sampling_mode=sampling_mode
    )
    
    # Visualize some predictions
    visualize_predictions(
        model, sparse_test_dataset, device, num_samples=5,
        save_path=os.path.join(args.save_dir, 'sample_predictions.png')
    )
    
    # Evaluate model performance at different sparsity levels
    sparsity_results = evaluate_by_sparsity(
        model, sparse_test_dataset, device, 
        sparsity_levels=args.sparsity_levels,
        sampling_mode=sampling_mode
    )
    
    # Plot and save sparsity results
    plot_sparsity_results(
        sparsity_results,
        save_path=os.path.join(args.save_dir, 'sparsity_vs_accuracy.png')
    )
    
    # Save results to file
    with open(os.path.join(args.save_dir, 'sparsity_results.txt'), 'w') as f:
        f.write("Number of Pixels,Accuracy\n")
        for pixels, accuracy in sorted(sparsity_results.items()):
            f.write(f"{pixels},{accuracy:.2f}\n")

if __name__ == '__main__':
    main() 