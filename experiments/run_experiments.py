import torch
import numpy as np
import random
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from judge.judge_model import JudgeModel
from judge.data_utils import get_mnist_test_dataset
from .evaluate_debate import simulate_debate
import itertools

def run_comprehensive_experiment(num_games_per_condition=100, num_simulations=250, save_dir='comprehensive_results'):
    """
    Run comprehensive experiment across all variations:
    - 2 sampling methods: nonzero, weighted (with matching judge models)
    - 3 pixel counts: 4, 6, 8
    - 2 precommit settings: True, False
    Total: 12 conditions (6 per sampling method)
    """
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Checkpoint file for resuming
    checkpoint_file = os.path.join(experiment_dir, 'checkpoint.json')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define judge models for each sampling method (hardcoded for matching)
    judge_models = {
        'nonzero': 'models_cnn/best_model_cnn_nonzero_8px.pth',
        'weighted': 'models_cnn/best_model_cnn_weighted_8px.pth'
    }
    
    # Load and validate all judge models
    loaded_judge_models = {}
    for sampling_method, model_path in judge_models.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Judge model not found: {model_path}")
        
        judge_model = JudgeModel(model_path, device=device)
        if judge_model.model_type != 'cnn':
            raise ValueError(f"Model {model_path} is not a CNN model")
        
        # Verify the model was trained on the expected sampling method
        if judge_model.sampling_mode != sampling_method:
            print(f"Warning: Model {model_path} was trained on '{judge_model.sampling_mode}' but expected '{sampling_method}'")
        
        # Increase batch size for faster neural network evaluations
        if hasattr(judge_model, 'batch_size'):
            judge_model.batch_size = 256  # or even 512 for 4GB GPU
        
        loaded_judge_models[sampling_method] = judge_model
        print(f"Loaded {sampling_method} judge model: {model_path} (trained on {judge_model.sampling_mode})")
    
    # Load test dataset
    test_dataset = get_mnist_test_dataset('./data')
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Define experimental conditions
    sampling_methods = ['nonzero', 'weighted']  # Each uses its matching judge model
    pixel_counts = [4, 6, 8]
    precommit_settings = [True, False]
    
    # Create all combinations
    conditions = list(itertools.product(sampling_methods, pixel_counts, precommit_settings))
    
    # Check for existing checkpoint to resume from
    completed_conditions = set()
    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            comprehensive_results = json.load(f)
        completed_conditions = set(comprehensive_results['conditions'].keys())
        print(f"Resuming experiment - {len(completed_conditions)} conditions already completed")
    else:
        # Storage for all results (new experiment)
        comprehensive_results = {
            'experiment_info': {
                'timestamp': timestamp,
                'judge_models': judge_models,
                'num_games_per_condition': num_games_per_condition,
                'num_simulations': num_simulations,
                'total_conditions': len(conditions),
                'total_games': len(conditions) * num_games_per_condition
            },
            'conditions': {},
            'summary_stats': {}
        }
    
    print(f"\nRunning {len(conditions)} conditions with {num_games_per_condition} games each")
    print(f"Total games: {len(conditions) * num_games_per_condition}")
    
    # Sample games once for all conditions to ensure fair comparison
    game_indices = np.random.choice(len(test_dataset), num_games_per_condition, replace=False).tolist()
    
    # Before the condition loop, pre-load all images
    print("Pre-loading game images...")
    cached_games = []
    for game_idx in game_indices:
        image, true_label = test_dataset[game_idx]
        image_np = image.squeeze().numpy()
        cached_games.append((game_idx, image_np, int(true_label)))
    
    # Run experiments for each condition
    for condition_idx, (sampling_method, pixel_count, precommit) in enumerate(conditions):
        condition_name = f"{sampling_method}_{pixel_count}px_precommit{precommit}"
        if condition_name in completed_conditions:
            print(f"\n[{condition_idx + 1}/{len(conditions)}] Condition already completed: {condition_name}")
            continue
        
        print(f"\n[{condition_idx + 1}/{len(conditions)}] Running condition: {condition_name}")
        
        # Get the appropriate judge model for this sampling method
        judge_model = loaded_judge_models[sampling_method]
        
        condition_results = {
            'sampling_method': sampling_method,
            'pixel_count': pixel_count,
            'precommit': precommit,
            'judge_model_path': judge_models[sampling_method],
            'games': [],
            'stats': {
                'judge_picks_true': 0,
                'judge_does_not_pick_true': 0,
                'judge_picks_deceptive': 0,  # Only applicable for precommit=True
                'judge_picks_honest_over_deceptive': 0,  # Only applicable for precommit=True
            }
        }
        
        # Run games for this condition
        for game_idx, image_np, true_label in cached_games:
            game_result = simulate_debate(
                judge_model=judge_model,
                image=image_np,
                true_label=true_label,
                total_pixels=pixel_count,
                num_simulations=num_simulations,
                temperature=0.5,
                verbose=False,
                sampling_mode=sampling_method,
                precommit=precommit,
                visualize=False,
                save_dir=None
            )
            
            # Extract game data for future judge training
            game_data = {
                'game_idx': int(game_idx),
                'true_label': int(game_result['true_label']),
                'deception_target_label': int(game_result['deception_target_label']),
                'predicted_digit': int(game_result['predicted_digit']),
                'revealed_pixels': [(int(x), int(y), float(value)) for x, y, value in game_result['revealed_pixels']],  # Convert to native types
                'move_sequence': [],  # Will populate with player info
                'final_probabilities': game_result['probabilities'].tolist(),
                'true_prob': float(game_result['true_prob']),
                'deception_target_prob': float(game_result['deception_target_prob']),
                'honest_won': bool(game_result['honest_won']),
                'deceptive_won': bool(game_result['deceptive_won'])
            }
            
            # Add move sequence with player information
            for move_idx, (x, y, value) in enumerate(game_result['revealed_pixels']):
                player_type = 'honest' if move_idx % 2 == 0 else 'deceptive'
                game_data['move_sequence'].append({
                    'move_number': int(move_idx + 1),
                    'player': player_type,
                    'pixel_position': [int(x), int(y)],
                    'pixel_value': float(value)
                })
            
            condition_results['games'].append(game_data)
            
            # Update statistics (convert to Python int to avoid JSON serialization issues)
            condition_results['stats']['judge_picks_true'] += int(game_result['predicted_digit'] == game_result['true_label'])
            condition_results['stats']['judge_does_not_pick_true'] += int(game_result['predicted_digit'] != game_result['true_label'])
            
            if precommit:
                # Judge picks deceptive target
                condition_results['stats']['judge_picks_deceptive'] += int(game_result['predicted_digit'] == game_result['deception_target_label'])
                
                # Judge picks honest over deceptive (comparing probabilities/logits)
                condition_results['stats']['judge_picks_honest_over_deceptive'] += int(
                    game_result['true_prob'] > game_result['deception_target_prob']
                )
            
        
        # Calculate percentages
        total_games = len(condition_results['games'])
        condition_results['stats']['judge_picks_true_rate'] = (
            condition_results['stats']['judge_picks_true'] / total_games * 100
        )
        condition_results['stats']['judge_does_not_pick_true_rate'] = (
            condition_results['stats']['judge_does_not_pick_true'] / total_games * 100
        )
        
        if precommit:
            condition_results['stats']['judge_picks_deceptive_rate'] = (
                condition_results['stats']['judge_picks_deceptive'] / total_games * 100
            )
            condition_results['stats']['judge_picks_honest_over_deceptive_rate'] = (
                condition_results['stats']['judge_picks_honest_over_deceptive'] / total_games * 100
            )
        else:
            condition_results['stats']['judge_picks_deceptive_rate'] = None
            condition_results['stats']['judge_picks_honest_over_deceptive_rate'] = None
        
        # Store condition results
        comprehensive_results['conditions'][condition_name] = condition_results
        
        # Save checkpoint after each condition
        with open(checkpoint_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        print(f"  Checkpoint saved: {condition_name} completed")
        
        # Print condition summary
        print(f"  Judge picks true: {condition_results['stats']['judge_picks_true_rate']:.1f}%")
        print(f"  Judge does not pick true: {condition_results['stats']['judge_does_not_pick_true_rate']:.1f}%")
        if precommit:
            print(f"  Judge picks deceptive: {condition_results['stats']['judge_picks_deceptive_rate']:.1f}%")
            print(f"  Judge picks honest over deceptive (prob): {condition_results['stats']['judge_picks_honest_over_deceptive_rate']:.1f}%")
    
    # Calculate summary statistics across all conditions
    comprehensive_results['summary_stats'] = calculate_summary_stats(comprehensive_results['conditions'])
    
    # Save comprehensive results
    results_file = os.path.join(experiment_dir, 'comprehensive_results.json')
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save summary report
    summary_file = os.path.join(experiment_dir, 'summary_report.txt')
    write_summary_report(comprehensive_results, summary_file)
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {experiment_dir}")
    print(f"Summary report: {summary_file}")
    print(f"Full data: {results_file}")
    
    return comprehensive_results, experiment_dir

def calculate_summary_stats(conditions):
    """Calculate summary statistics across all conditions"""
    summary = {
        'by_sampling_method': {},
        'by_pixel_count': {},
        'by_precommit': {},
        'overall': {
            'total_games': 0,
            'avg_judge_picks_true_rate': 0,
            'avg_judge_does_not_pick_true_rate': 0,
            'avg_judge_picks_deceptive_rate': 0,
            'avg_judge_picks_honest_over_deceptive_rate': 0
        }
    }
    
    # Group by different factors
    for condition_name, condition_data in conditions.items():
        sampling_method = condition_data['sampling_method']
        pixel_count = condition_data['pixel_count']
        precommit = condition_data['precommit']
        stats = condition_data['stats']
        
        # Initialize dictionaries if needed
        for key in [sampling_method, pixel_count, precommit]:
            for group_dict in [summary['by_sampling_method'], summary['by_pixel_count'], summary['by_precommit']]:
                if (key == sampling_method and group_dict == summary['by_sampling_method']) or \
                   (key == pixel_count and group_dict == summary['by_pixel_count']) or \
                   (key == precommit and group_dict == summary['by_precommit']):
                    if key not in group_dict:
                        group_dict[key] = {
                            'conditions': 0,
                            'total_games': 0,
                            'judge_picks_true_rate': 0,
                            'judge_does_not_pick_true_rate': 0,
                            'judge_picks_deceptive_rate': 0,
                            'judge_picks_honest_over_deceptive_rate': 0
                        }
        
        # Update groupings
        num_games = len(condition_data['games'])
        
        # Update by sampling method
        summary['by_sampling_method'][sampling_method]['conditions'] += 1
        summary['by_sampling_method'][sampling_method]['total_games'] += num_games
        summary['by_sampling_method'][sampling_method]['judge_picks_true_rate'] += stats['judge_picks_true_rate']
        summary['by_sampling_method'][sampling_method]['judge_does_not_pick_true_rate'] += stats['judge_does_not_pick_true_rate']
        if stats['judge_picks_deceptive_rate'] is not None:
            summary['by_sampling_method'][sampling_method]['judge_picks_deceptive_rate'] += stats['judge_picks_deceptive_rate']
        if stats['judge_picks_honest_over_deceptive_rate'] is not None:
            summary['by_sampling_method'][sampling_method]['judge_picks_honest_over_deceptive_rate'] += stats['judge_picks_honest_over_deceptive_rate']
        
        # Update by pixel count
        summary['by_pixel_count'][pixel_count]['conditions'] += 1
        summary['by_pixel_count'][pixel_count]['total_games'] += num_games
        summary['by_pixel_count'][pixel_count]['judge_picks_true_rate'] += stats['judge_picks_true_rate']
        summary['by_pixel_count'][pixel_count]['judge_does_not_pick_true_rate'] += stats['judge_does_not_pick_true_rate']
        if stats['judge_picks_deceptive_rate'] is not None:
            summary['by_pixel_count'][pixel_count]['judge_picks_deceptive_rate'] += stats['judge_picks_deceptive_rate']
        if stats['judge_picks_honest_over_deceptive_rate'] is not None:
            summary['by_pixel_count'][pixel_count]['judge_picks_honest_over_deceptive_rate'] += stats['judge_picks_honest_over_deceptive_rate']
        
        # Update by precommit
        summary['by_precommit'][precommit]['conditions'] += 1
        summary['by_precommit'][precommit]['total_games'] += num_games
        summary['by_precommit'][precommit]['judge_picks_true_rate'] += stats['judge_picks_true_rate']
        summary['by_precommit'][precommit]['judge_does_not_pick_true_rate'] += stats['judge_does_not_pick_true_rate']
        if stats['judge_picks_deceptive_rate'] is not None:
            summary['by_precommit'][precommit]['judge_picks_deceptive_rate'] += stats['judge_picks_deceptive_rate']
        if stats['judge_picks_honest_over_deceptive_rate'] is not None:
            summary['by_precommit'][precommit]['judge_picks_honest_over_deceptive_rate'] += stats['judge_picks_honest_over_deceptive_rate']
        
        # Update overall
        summary['overall']['total_games'] += num_games
        summary['overall']['avg_judge_picks_true_rate'] += stats['judge_picks_true_rate']
        summary['overall']['avg_judge_does_not_pick_true_rate'] += stats['judge_does_not_pick_true_rate']
        if stats['judge_picks_deceptive_rate'] is not None:
            summary['overall']['avg_judge_picks_deceptive_rate'] += stats['judge_picks_deceptive_rate']
        if stats['judge_picks_honest_over_deceptive_rate'] is not None:
            summary['overall']['avg_judge_picks_honest_over_deceptive_rate'] += stats['judge_picks_honest_over_deceptive_rate']
    
    # Calculate averages
    num_conditions = len(conditions)
    summary['overall']['avg_judge_picks_true_rate'] /= num_conditions
    summary['overall']['avg_judge_does_not_pick_true_rate'] /= num_conditions
    
    # Count precommit conditions for correct averaging
    precommit_conditions = sum(1 for _, data in conditions.items() if data['precommit'])
    if precommit_conditions > 0:
        summary['overall']['avg_judge_picks_deceptive_rate'] /= precommit_conditions
        summary['overall']['avg_judge_picks_honest_over_deceptive_rate'] /= precommit_conditions
    
    # Calculate averages for grouped statistics
    for group_name, group_data in [('by_sampling_method', summary['by_sampling_method']),
                                   ('by_pixel_count', summary['by_pixel_count']),
                                   ('by_precommit', summary['by_precommit'])]:
        for key, data in group_data.items():
            if data['conditions'] > 0:
                data['judge_picks_true_rate'] /= data['conditions']
                data['judge_does_not_pick_true_rate'] /= data['conditions']
                
                # Only divide by conditions that have precommit=True for these metrics
                if group_name == 'by_precommit' and key == True:
                    data['judge_picks_deceptive_rate'] /= data['conditions']
                    data['judge_picks_honest_over_deceptive_rate'] /= data['conditions']
                elif group_name != 'by_precommit':
                    # For sampling method and pixel count, count only precommit conditions
                    precommit_count = sum(1 for _, cond_data in conditions.items() 
                                        if ((group_name == 'by_sampling_method' and cond_data['sampling_method'] == key) or
                                            (group_name == 'by_pixel_count' and cond_data['pixel_count'] == key)) and 
                                           cond_data['precommit'])
                    if precommit_count > 0:
                        data['judge_picks_deceptive_rate'] /= precommit_count
                        data['judge_picks_honest_over_deceptive_rate'] /= precommit_count
    
    return summary

def write_summary_report(results, output_file):
    """Write a human-readable summary report"""
    with open(output_file, 'w') as f:
        f.write("MNIST DEBATE EXPERIMENT COMPREHENSIVE RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Experiment info
        info = results['experiment_info']
        f.write(f"Experiment timestamp: {info['timestamp']}\n")
        f.write(f"Judge models used:\n")
        for sampling_method, model_path in info['judge_models'].items():
            f.write(f"  {sampling_method}: {model_path}\n")
        f.write(f"Games per condition: {info['num_games_per_condition']}\n")
        f.write(f"MCTS simulations per move: {info['num_simulations']}\n")
        f.write(f"Total conditions: {info['total_conditions']}\n")
        f.write(f"Total games: {info['total_games']}\n\n")
        
        # Overall results
        overall = results['summary_stats']['overall']
        f.write("OVERALL RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average judge picks true digit: {overall['avg_judge_picks_true_rate']:.1f}%\n")
        f.write(f"Average judge does not pick true digit: {overall['avg_judge_does_not_pick_true_rate']:.1f}%\n")
        f.write(f"Average judge picks deceptive (precommit cases): {overall['avg_judge_picks_deceptive_rate']:.1f}%\n")
        f.write(f"Average judge picks honest over deceptive (precommit cases): {overall['avg_judge_picks_honest_over_deceptive_rate']:.1f}%\n\n")
        
        # Results by sampling method
        f.write("RESULTS BY SAMPLING METHOD\n")
        f.write("-" * 30 + "\n")
        for method, stats in results['summary_stats']['by_sampling_method'].items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Judge picks true: {stats['judge_picks_true_rate']:.1f}%\n")
            f.write(f"  Judge does not pick true: {stats['judge_does_not_pick_true_rate']:.1f}%\n")
            f.write(f"  Judge picks deceptive (precommit): {stats['judge_picks_deceptive_rate']:.1f}%\n")
            f.write(f"  Judge picks honest over deceptive (precommit): {stats['judge_picks_honest_over_deceptive_rate']:.1f}%\n\n")
        
        # Results by pixel count
        f.write("RESULTS BY PIXEL COUNT\n")
        f.write("-" * 25 + "\n")
        for pixels, stats in results['summary_stats']['by_pixel_count'].items():
            f.write(f"{pixels} PIXELS:\n")
            f.write(f"  Judge picks true: {stats['judge_picks_true_rate']:.1f}%\n")
            f.write(f"  Judge does not pick true: {stats['judge_does_not_pick_true_rate']:.1f}%\n")
            f.write(f"  Judge picks deceptive (precommit): {stats['judge_picks_deceptive_rate']:.1f}%\n")
            f.write(f"  Judge picks honest over deceptive (precommit): {stats['judge_picks_honest_over_deceptive_rate']:.1f}%\n\n")
        
        # Results by precommit
        f.write("RESULTS BY PRECOMMIT SETTING\n")
        f.write("-" * 30 + "\n")
        for precommit, stats in results['summary_stats']['by_precommit'].items():
            f.write(f"PRECOMMIT {precommit}:\n")
            f.write(f"  Judge picks true: {stats['judge_picks_true_rate']:.1f}%\n")
            f.write(f"  Judge does not pick true: {stats['judge_does_not_pick_true_rate']:.1f}%\n")
            if precommit:
                f.write(f"  Judge picks deceptive: {stats['judge_picks_deceptive_rate']:.1f}%\n")
                f.write(f"  Judge picks honest over deceptive: {stats['judge_picks_honest_over_deceptive_rate']:.1f}%\n")
            f.write("\n")
        
        # Detailed condition results
        f.write("DETAILED CONDITION RESULTS\n")
        f.write("-" * 30 + "\n")
        for condition_name, condition_data in results['conditions'].items():
            f.write(f"{condition_name.upper()}:\n")
            stats = condition_data['stats']
            f.write(f"  Judge picks true: {stats['judge_picks_true_rate']:.1f}%\n")
            f.write(f"  Judge does not pick true: {stats['judge_does_not_pick_true_rate']:.1f}%\n")
            if condition_data['precommit']:
                f.write(f"  Judge picks deceptive: {stats['judge_picks_deceptive_rate']:.1f}%\n")
                f.write(f"  Judge picks honest over deceptive (prob): {stats['judge_picks_honest_over_deceptive_rate']:.1f}%\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive MNIST debate experiment')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games per condition')
    parser.add_argument('--num-simulations', type=int, default=250,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--save-dir', type=str, default='comprehensive_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run comprehensive experiment
    results, experiment_dir = run_comprehensive_experiment(
        num_games_per_condition=args.num_games,
        num_simulations=args.num_simulations,
        save_dir=args.save_dir
    )
    
    print(f"\nExperiment completed successfully!")
    print(f"Results directory: {experiment_dir}")

if __name__ == '__main__':
    main() 