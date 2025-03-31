import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.patches import Patch
from judge_model import create_sparse_image

def visualize_debate_sequence(result, judge_model=None, save_path=None, show=True):
    """
    Visualize the sequence of moves in a debate game, showing how the judge's probabilities evolve.
    
    Args:
        result: Dictionary containing game results
        judge_model: The judge model used for the debate
        save_path: Path to save the visualization (if None, will display instead)
        show: Whether to display the plot
    """
    # Extract data from game result
    revealed_pixels = result['revealed_pixels']
    true_label = result['true_label']
    deception_target_label = result['deception_target_label']
    
    # Create the full image to reveal
    full_image = np.zeros((28, 28), dtype=np.float32)
    
    # Fill in the revealed pixels
    for x, y, value in revealed_pixels:
        full_image[y, x] = value
    
    # Calculate number of steps (including initial state)
    num_steps = len(revealed_pixels) + 1
    
    # Create a figure with rows for each step and 2 columns
    fig, axes = plt.subplots(num_steps, 2, figsize=(12, 3 * num_steps))
    
    # For a single step, axes won't be a 2D array, so we need to handle that case
    if num_steps == 1:
        axes = np.array([axes])
    
    # Create color maps for the pixels
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Initialize revealed image and pixel colors
    revealed_image = np.zeros((28, 28))
    pixel_colors = np.zeros((28, 28))
    
    # Get initial prediction with no pixels revealed
    sparse_image = create_sparse_image([])
    predicted_digit, probabilities = judge_model.predict(sparse_image)
    
    # Plot initial state
    axes[0, 0].imshow(full_image, cmap='gray', alpha=0.6)  # Increased from 0.3 to 0.6
    axes[0, 0].imshow(revealed_image, cmap='gray', alpha=0.8)
    axes[0, 0].set_title('Initial State (No Pixels)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    # Plot initial probabilities
    bars = axes[0, 1].bar(range(10), probabilities)
    bars[true_label].set_color('blue')
    bars[deception_target_label].set_color('red')
    axes[0, 1].axhline(y=0.1, color='gray', linestyle='--', alpha=0.7)
    axes[0, 1].set_title(f'Initial Prediction: {predicted_digit}')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(range(10))
    
    # Plot each step
    for i, (x, y, value) in enumerate(revealed_pixels):
        # Update revealed image and pixel colors
        revealed_image[y, x] = value
        # 1 for honest player (even indices), 2 for deceptive player (odd indices)
        pixel_colors[y, x] = 1 if i % 2 == 0 else 2
        
        # Get prediction after this pixel is revealed
        sparse_image = create_sparse_image(revealed_pixels[:i+1])
        predicted_digit, probabilities = judge_model.predict(sparse_image)
        
        # Plot image with colored pixels
        axes[i+1, 0].imshow(full_image, cmap='gray', alpha=0.6)  # Increased from 0.3 to 0.6
        axes[i+1, 0].imshow(revealed_image, cmap='gray', alpha=0.8)
        axes[i+1, 0].imshow(pixel_colors, cmap=cmap, norm=norm, alpha=0.6)
        
        player = "Honest" if i % 2 == 0 else "Deceptive"
        axes[i+1, 0].set_title(f'Step {i+1}: {player} reveals ({x},{y})')
        axes[i+1, 0].set_xticks([])
        axes[i+1, 0].set_yticks([])
        
        # Plot probabilities
        bars = axes[i+1, 1].bar(range(10), probabilities)
        bars[true_label].set_color('blue')
        bars[deception_target_label].set_color('red')
        axes[i+1, 1].axhline(y=0.1, color='gray', linestyle='--', alpha=0.7)
        axes[i+1, 1].set_title(f'Prediction after Step {i+1}: {predicted_digit}')
        axes[i+1, 1].set_ylim(0, 1)
        axes[i+1, 1].set_xticks(range(10))
    
    # Add a legend to the last image
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Honest Player'),
        Patch(facecolor='red', edgecolor='red', label='Deceptive Player')
    ]
    axes[-1, 0].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        if not show:
            plt.close()
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_multiple_games(results_file, judge_model, num_games=5, save_dir=None, show=True):
    """
    Visualize multiple debate games from a results file.
    
    Args:
        results_file: Path to the JSON file containing debate results
        judge_model: The judge model used for the debates
        num_games: Number of games to visualize
        save_dir: Directory to save visualizations (if None, will display instead)
        show: Whether to display the plots
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get game results
    games = results['games']
    
    # Randomly select games to visualize
    if num_games > len(games):
        num_games = len(games)
    
    selected_indices = np.random.choice(len(games), num_games, replace=False)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Visualize each selected game
    for i, idx in enumerate(selected_indices):
        game = games[idx]
        
        # Convert probabilities back to numpy array if needed
        if isinstance(game['probabilities'], list):
            game['probabilities'] = np.array(game['probabilities'])
        
        # Handle the renamed field
        if 'target_label' in game and 'deception_target_label' not in game:
            game['deception_target_label'] = game['target_label']
        
        # Visualize sequence
        if save_dir:
            save_path = os.path.join(save_dir, f'game_{i+1}_sequence.png')
        else:
            save_path = None
        
        visualize_debate_sequence(game, judge_model, save_path=save_path, show=show) 