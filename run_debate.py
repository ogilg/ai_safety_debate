import torch
import numpy as np
import random
import json
import os
import argparse
from judge_model import JudgeModel
import matplotlib
# Use non-interactive backend if running without display
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
from data_utils import get_mnist_test_dataset
from evaluate_debate import evaluate_debate_performance

def main():
    parser = argparse.ArgumentParser(description='MNIST Debate Game with MCTS-PUCT')
    parser.add_argument('--model-path', type=str, default='models/best_model_cnn_random_10px.pth', 
                        help='Path to the trained judge model')
    parser.add_argument('--total-pixels', type=int, default=None, 
                        help='Total number of pixels to reveal (default: use judge model training value)')
    parser.add_argument('--num-simulations', type=int, default=100, 
                        help='Number of MCTS simulations per move')
    parser.add_argument('--precommit', action='store_true',
                        help='Whether the deceptive player has to precommit to a digit')
    parser.add_argument('--num-games', type=int, default=10, 
                        help='Number of games to evaluate')
    parser.add_argument('--save-dir', type=str, default='debate_results', 
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='Temperature for action selection (0=deterministic, higher=more random)')
    parser.add_argument('--sampling-mode', type=str, default=None,
                        choices=['random', 'nonzero'],
                        help='How to sample pixels: random or nonzero (default: use judge model training value)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the debate games')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for judge model evaluations')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load judge model
    judge_model = JudgeModel(args.model_path, device=device)
    
    # Extract num_pixels and sampling_mode from judge model if not specified
    if args.total_pixels is None:
        # Try to extract from model args
        model_args = judge_model.model_args if hasattr(judge_model, 'model_args') else {}
        total_pixels = model_args.get('num_pixels', 10)
        print(f"Using {total_pixels} total pixels from judge model training")
    else:
        total_pixels = args.total_pixels
    
    # Extract sampling_mode from judge model if not specified
    if args.sampling_mode is None:
        # Try to extract from model args
        model_args = judge_model.model_args if hasattr(judge_model, 'model_args') else {}
        sampling_mode = model_args.get('sampling_mode', 'random')
        print(f"Using '{sampling_mode}' sampling mode from judge model training")
    else:
        sampling_mode = args.sampling_mode
    
    # Load MNIST test dataset using data_utils
    test_dataset = get_mnist_test_dataset('./data')
    
    # Create visualization directory if needed
    viz_dir = os.path.join(args.save_dir, 'visualizations') if args.visualize else None
    
    # Evaluate debate performance with total_pixels and sampling_mode
    results = evaluate_debate_performance(
        judge_model,
        test_dataset,
        num_games=args.num_games,
        total_pixels=total_pixels,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        sampling_mode=sampling_mode,
        precommit=args.precommit,
        visualize=args.visualize,
        save_dir=viz_dir
    )
    
    # Save results
    with open(os.path.join(args.save_dir, 'debate_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for game in results['games']:
            # Remove full_image and revealed_pixels columns if they exist
            if 'full_image' in game:
                del game['full_image']
                del game['revealed_pixels']
            game['probabilities'] = game['probabilities'].tolist()
            
            # Convert numpy types within each game to Python native types
            for k, v in game.items():
                if isinstance(v, (np.bool_, np.integer, np.floating, np.float32)):
                    game[k] = v.item()
        
        # Convert any numpy types at the top level to Python native types for JSON serialization
        results = {k: v.item() if isinstance(v, (np.bool_, np.integer, np.floating, np.float32)) else v 
                  for k, v in results.items()}
        json.dump(results, f, indent=2)
    
    if args.visualize:
        print(f"Visualizations saved to {viz_dir}")

if __name__ == '__main__':
    main() 