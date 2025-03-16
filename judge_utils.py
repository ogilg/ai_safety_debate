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
        
        print(f"Loaded judge model from {model_path}")
        print(f"Model type: {model_type}")
        print(f"Sampling mode: {self.sampling_mode}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
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
        # Convert to tensor if numpy array
        if isinstance(sparse_image, np.ndarray):
            sparse_image = torch.FloatTensor(sparse_image)
        
        # Reshape based on model type
        if self.model_type == 'mlp':
            # Reshape to flat if needed
            if sparse_image.shape != (784,):
                sparse_image = sparse_image.reshape(-1)
        elif self.model_type == 'cnn':
            # Reshape to 2D if needed
            if sparse_image.shape != (1, 28, 28):
                if len(sparse_image.shape) == 1 or sparse_image.shape == (784,):
                    sparse_image = sparse_image.reshape(1, 28, 28)
                elif sparse_image.shape == (28, 28):
                    sparse_image = sparse_image.unsqueeze(0)  # Add channel dimension
        
        # Normalize if not already normalized
        if sparse_image.max() > 1.0:
            sparse_image = sparse_image / 255.0
        
        # Apply MNIST normalization
        sparse_image = (sparse_image - 0.1307) / 0.3081
        
        # Make prediction
        with torch.no_grad():
            input_tensor = sparse_image.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_digit = output.argmax(dim=1).item()
        
        return predicted_digit, probabilities.cpu().numpy()
    
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