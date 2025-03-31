import torch
import numpy as np
from train_judge import MNISTJudge

class JudgeModel:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_args = checkpoint['args']
        
        # Store model args for reference
        self.model_args = model_args
        
        # Store number of pixels the model was trained on
        self.num_pixels = model_args.get('num_pixels', 10)
        
        # Get model type and sampling mode (default to 'mlp' and 'random' for backward compatibility)
        model_type = model_args.get('model_type', 'mlp')
        self.sampling_mode = model_args.get('sampling_mode', 'random')
        
        self.model = MNISTJudge(
            input_size=784,
            hidden_sizes=model_args['hidden_sizes'],
            output_size=10,
            dropout_rate=0.0,  # No dropout during inference
            model_type=model_type
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.model_type = model_type
        
        # Constants for MNIST normalization
        self.norm_mean = 0.1307
        self.norm_std = 0.3081
        
        print(f"Loaded judge model from {model_path}")
        print(f"Model type: {model_type}")
        print(f"Sampling mode: {self.sampling_mode}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    def _preprocess_image(self, sparse_image):
        """
        Preprocess a single sparse image dictionary for model input
        
        Args:
            sparse_image: Dictionary with 'indices' and 'values' keys
            
        Returns:
            Processed tensor ready for model input
        """
        # Create empty tensor based on model type
        if self.model_type == 'mlp':
            # Create flat tensor of zeros
            image = torch.zeros(784, dtype=torch.float32, device=self.device)
            
            # Fill in the revealed pixels
            for idx, val in zip(sparse_image['indices'], sparse_image['values']):
                image[idx] = val
                
        else:  # self.model_type == 'cnn'
            # Create 2D tensor of zeros
            image = torch.zeros((1, 28, 28), dtype=torch.float32, device=self.device)
            
            # Fill in the revealed pixels
            for idx, val in zip(sparse_image['indices'], sparse_image['values']):
                # Convert flat index to 2D coordinates
                y, x = idx // 28, idx % 28
                image[0, y, x] = val
        
        return image
    
    def _preprocess_batch(self, sparse_images):
        """
        Preprocess a batch of sparse image dictionaries
        
        Args:
            sparse_images: List of dictionaries, each with 'indices' and 'values' keys
            
        Returns:
            Processed batch tensor ready for model input
        """
        # Handle empty batch
        if len(sparse_images) == 0:
            return torch.tensor([])
        
        # For MLP models, we can use a more efficient batch preprocessing
        if self.model_type == 'mlp':
            # Create batch tensor of zeros
            batch_tensor = torch.zeros((len(sparse_images), 784), dtype=torch.float32, device=self.device)
            
            # Fill in revealed pixels for each image
            for i, sparse_img in enumerate(sparse_images):
                for idx, val in zip(sparse_img['indices'], sparse_img['values']):
                    batch_tensor[i, idx] = val
        
        # For CNN models
        else:  # self.model_type == 'cnn'
            # Create batch tensor of zeros
            batch_tensor = torch.zeros((len(sparse_images), 1, 28, 28), dtype=torch.float32, device=self.device)
            
            # Fill in revealed pixels for each image
            for i, sparse_img in enumerate(sparse_images):
                for idx, val in zip(sparse_img['indices'], sparse_img['values']):
                    # Convert flat index to 2D coordinates
                    y, x = idx // 28, idx % 28
                    batch_tensor[i, 0, y, x] = val
        
        return batch_tensor
    
    def _forward_pass(self, input_tensor):
        """
        Perform forward pass through the model
        
        Args:
            input_tensor: Preprocessed tensor ready for model inference
            
        Returns:
            predicted_digits: Predicted digit classes
            probabilities: Softmax probabilities for each class
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_digits = outputs.argmax(dim=1)
            
        return predicted_digits, probabilities
    
    @torch.no_grad()
    def predict(self, sparse_image):
        """
        Make a prediction based on a sparse image dictionary
        
        Args:
            sparse_image: Dictionary with 'indices' and 'values' keys
        
        Returns:
            predicted_digit: The predicted digit (0-9)
            probabilities: Probability distribution over all digits
        """
        # Preprocess the image
        processed_image = self._preprocess_image(sparse_image)
        
        # Add batch dimension if needed (for MLP model)
        if self.model_type == 'mlp':
            processed_image = processed_image.unsqueeze(0)
        
        # Forward pass
        predicted_digits, probabilities = self._forward_pass(processed_image)
        
        # Return single result
        return predicted_digits[0].item(), probabilities[0].cpu().numpy()
    
    @torch.no_grad()
    def batch_predict(self, sparse_images):
        """
        Make predictions on a batch of sparse image dictionaries
        
        Args:
            sparse_images: List of dictionaries, each with 'indices' and 'values' keys
        
        Returns:
            predicted_digits: Array of predicted digits (0-9)
            probabilities: Array of probability distributions over all digits
        """
        batch_size = len(sparse_images)
        if batch_size == 0:
            return [], []
        
        # Preprocess batch at once
        batch_tensor = self._preprocess_batch(sparse_images)
        
        # Forward pass
        predicted_digits, probabilities = self._forward_pass(batch_tensor)
        
        return predicted_digits.cpu().numpy(), probabilities.cpu().numpy()
    
    def get_confidence(self, sparse_image, digit=None):
        """
        Get the model's confidence in a specific digit or the predicted digit
        
        Args:
            sparse_image: A tensor or numpy array of shape (784,) or (28, 28)
            digit: Optional, the digit to get confidence for. If None, returns
                   confidence in the predicted digit.
        
        Returns:
            confidence: Confidence value (0-1)
            predicted_digit: The predicted digit
        """
        predicted_digit, probabilities = self.predict(sparse_image)
        
        if digit is None:
            confidence = probabilities[predicted_digit]
        else:
            confidence = probabilities[digit]
        
        return confidence, predicted_digit

def create_sparse_image(revealed_pixels):
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