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
    
    def _preprocess_image(self, image):
        """
        Preprocess a single image for model input
        
        Args:
            image: A numpy array or torch tensor (28x28 or 784)
            
        Returns:
            Processed tensor ready for model input
        """
        # Convert to tensor if numpy array
        if isinstance(image, np.ndarray):
            image = torch.FloatTensor(image)
        
        # Reshape based on model type
        if self.model_type == 'mlp':
            # Reshape to flat if needed
            if image.shape != (784,):
                image = image.reshape(-1)
        elif self.model_type == 'cnn':
            # Reshape to 2D if needed
            if image.shape != (1, 28, 28):
                if len(image.shape) == 1 or image.shape == (784,):
                    image = image.reshape(1, 28, 28)
                elif image.shape == (28, 28):
                    image = image.unsqueeze(0)  # Add channel dimension

        return image
    
    def _preprocess_batch(self, images):
        """
        Preprocess a batch of images for model input using vectorized operations
        
        Args:
            images: A list of numpy arrays or torch tensors
            
        Returns:
            Processed batch tensor ready for model input
        """
        # Handle empty batch
        if len(images) == 0:
            return torch.tensor([])
        
        # Convert list to tensors if needed
        tensor_list = []
        for img in images:
            if isinstance(img, np.ndarray):
                tensor_list.append(torch.FloatTensor(img))
            else:
                tensor_list.append(img)
        
        # Process based on model type
        if self.model_type == 'mlp':
            # Ensure all images are flattened
            for i, img in enumerate(tensor_list):
                if img.shape != (784,):
                    tensor_list[i] = img.reshape(-1)
            
            # Stack all flattened images
            batch_tensor = torch.stack(tensor_list)
            
        elif self.model_type == 'cnn':
            # Ensure all images have shape (1, 28, 28)
            for i, img in enumerate(tensor_list):
                if img.shape != (1, 28, 28):
                    if len(img.shape) == 1 or img.shape == (784,):
                        tensor_list[i] = img.reshape(1, 28, 28)
                    elif img.shape == (28, 28):
                        tensor_list[i] = img.unsqueeze(0)
            
            # Stack all images
            batch_tensor = torch.stack(tensor_list)
        
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
    
    def predict(self, sparse_image):
        """
        Make a prediction based on a sparse image
        
        Args:
            sparse_image: A tensor or numpy array of shape (784,) or (28, 28)
                        with most values being 0 (hidden) and some values being
                        the actual pixel values (revealed)
        
        Returns:
            predicted_digit: The predicted digit (0-9)
            probabilities: Probability distribution over all digits
        """
        # Preprocess the image
        processed_image = self._preprocess_image(sparse_image)
        
        # Add batch dimension if not present
        if processed_image.dim() == 1 or (self.model_type == 'cnn' and processed_image.dim() == 3):
            processed_image = processed_image.unsqueeze(0)
        
        # Forward pass
        predicted_digits, probabilities = self._forward_pass(processed_image)
        
        # Return single result
        return predicted_digits[0].item(), probabilities[0].cpu().numpy()
    
    def batch_predict(self, sparse_images):
        """
        Make predictions on a batch of sparse images using vectorized operations
        
        Args:
            sparse_images: A list of sparse images (numpy arrays or tensors)
        
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

def create_sparse_image(revealed_pixels, image_size=(28, 28)):
    """
    Create a sparse image with only the revealed pixels
    
    Args:
        revealed_pixels: List of tuples [(x1, y1, val1), (x2, y2, val2), ...]
                         where (x, y) are coordinates and val is the pixel value
        image_size: Size of the output image
    
    Returns:
        sparse_image: A numpy array with revealed pixels set to their values
    """
    sparse_image = np.zeros(image_size, dtype=np.float32)
    
    for x, y, val in revealed_pixels:
        sparse_image[y, x] = val
    
    return sparse_image 