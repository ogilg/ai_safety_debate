import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_experiment_results(results_path):
    """Load experiment results from JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def create_analysis_dataframe(results):
    """Convert results into a pandas DataFrame for easier analysis"""
    data = []
    
    for condition_name, condition_data in results['conditions'].items():
        stats = condition_data['stats']
        data.append({
            'condition_name': condition_name,
            'sampling_method': condition_data['sampling_method'],
            'pixel_count': condition_data['pixel_count'],
            'precommit': condition_data['precommit'],
            'judge_picks_true_rate': stats['judge_picks_true_rate'],
            'judge_does_not_pick_true_rate': stats['judge_does_not_pick_true_rate'],
            'judge_picks_deceptive_rate': stats.get('judge_picks_deceptive_rate'),
            'judge_picks_honest_over_deceptive_rate': stats.get('judge_picks_honest_over_deceptive_rate'),
            'num_games': len(condition_data['games'])
        })
    
    return pd.DataFrame(data)

def plot_main_results(df, save_dir):
    """Create main visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MNIST Debate Experiment Results', fontsize=16, y=0.98)
    
    # Graph A: Judge picks true across all conditions
    
    # A1. Judge picks true by sampling method
    sampling_data = df.groupby('sampling_method')['judge_picks_true_rate'].mean()
    axes[0, 0].bar(sampling_data.index, sampling_data.values, color=['skyblue', 'lightgreen'])
    axes[0, 0].set_title('Judge Picks True Digit\nby Sampling Method')
    axes[0, 0].set_ylabel('Percentage (%)')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(sampling_data.values):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # A2. Judge picks true by pixel count
    pixel_data = df.groupby('pixel_count')['judge_picks_true_rate'].mean()
    axes[0, 1].bar(pixel_data.index.astype(str), pixel_data.values, color=['orange', 'gold', 'lightpink'])
    axes[0, 1].set_title('Judge Picks True Digit\nby Pixel Count')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_xlabel('Number of Pixels')
    axes[0, 1].set_ylim(0, 100)
    for i, v in enumerate(pixel_data.values):
        axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # A3. Judge picks true by precommit setting
    precommit_data = df.groupby('precommit')['judge_picks_true_rate'].mean()
    colors = ['lightblue', 'lightgreen']
    axes[0, 2].bar(['No Precommit', 'Precommit'], precommit_data.values, color=colors)
    axes[0, 2].set_title('Judge Picks True Digit\nby Precommit Setting')
    axes[0, 2].set_ylabel('Percentage (%)')
    axes[0, 2].set_ylim(0, 100)
    for i, v in enumerate(precommit_data.values):
        axes[0, 2].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Graph B: Precommit=True cases with three metrics
    
    # Filter for precommit=True conditions
    precommit_df = df[df['precommit'] == True]
    
    if len(precommit_df) > 0:
        # B1. All three metrics by sampling method (precommit=True only)
        sampling_methods = precommit_df['sampling_method'].unique()
        metrics = ['judge_picks_true_rate', 'judge_picks_deceptive_rate', 'judge_picks_honest_over_deceptive_rate']
        metric_labels = ['Judge Picks True', 'Judge Picks Deceptive', 'Judge Picks Honest > Deceptive']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        x = np.arange(len(sampling_methods))
        width = 0.25
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [precommit_df[precommit_df['sampling_method'] == method][metric].mean() 
                     for method in sampling_methods]
            axes[1, 0].bar(x + i*width, values, width, label=label, color=color)
        
        axes[1, 0].set_title('Precommit=True: All Metrics\nby Sampling Method')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_xlabel('Sampling Method')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(sampling_methods)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 100)
        
        # B2. All three metrics by pixel count (precommit=True only)
        pixel_counts = sorted(precommit_df['pixel_count'].unique())
        
        x = np.arange(len(pixel_counts))
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [precommit_df[precommit_df['pixel_count'] == pixels][metric].mean() 
                     for pixels in pixel_counts]
            axes[1, 1].bar(x + i*width, values, width, label=label, color=color)
        
        axes[1, 1].set_title('Precommit=True: All Metrics\nby Pixel Count')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_xlabel('Number of Pixels')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels([f'{p}px' for p in pixel_counts])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 100)
        
        # B3. Overall comparison of the three metrics (precommit=True only)
        overall_values = [precommit_df[metric].mean() for metric in metrics]
        bars = axes[1, 2].bar(metric_labels, overall_values, color=colors)
        axes[1, 2].set_title('Precommit=True: Overall\nMetric Comparison')
        axes[1, 2].set_ylabel('Percentage (%)')
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, value + 1, 
                           f'{value:.1f}%', ha='center', va='bottom')
    
    else:
        # If no precommit data, show message
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'No Precommit=True Data Available', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_xlim(0, 1)
            axes[1, i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmaps(df, save_dir):
    """Create heatmap visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MNIST Debate Results Heatmaps', fontsize=16)
    
    # 1. Judge picks true - Sampling method vs Pixel count
    pivot1 = df.pivot_table(values='judge_picks_true_rate', 
                           index='sampling_method', 
                           columns='pixel_count', 
                           aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Judge Picks True Rate (%)\nSampling Method vs Pixel Count')
    
    # 2. Honest win rate - Sampling method vs Pixel count
    pivot2 = df.pivot_table(values='honest_win_rate', 
                           index='sampling_method', 
                           columns='pixel_count', 
                           aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Honest Win Rate (%)\nSampling Method vs Pixel Count')
    
    # 3. Judge picks true - Precommit vs everything else
    pivot3 = df.pivot_table(values='judge_picks_true_rate', 
                           index='precommit', 
                           columns=['sampling_method', 'pixel_count'], 
                           aggfunc='mean')
    sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='Greens', ax=axes[1, 0])
    axes[1, 0].set_title('Judge Picks True Rate (%)\nPrecommit vs Conditions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Deceptive win rate - Sampling method vs Pixel count
    pivot4 = df.pivot_table(values='deceptive_win_rate', 
                           index='sampling_method', 
                           columns='pixel_count', 
                           aggfunc='mean')
    sns.heatmap(pivot4, annot=True, fmt='.1f', cmap='Reds', ax=axes[1, 1])
    axes[1, 1].set_title('Deceptive Win Rate (%)\nSampling Method vs Pixel Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precommit_analysis(df, save_dir):
    """Analyze precommit-specific results"""
    
    # Filter precommit conditions
    precommit_df = df[df['precommit'] == True]
    
    if len(precommit_df) == 0:
        print("No precommit data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Precommit Condition Analysis', fontsize=16)
    
    # 1. Judge picks honest over deceptive by sampling method
    if 'judge_picks_honest_over_deceptive_rate' in precommit_df.columns:
        honest_deceptive_sampling = precommit_df.groupby('sampling_method')['judge_picks_honest_over_deceptive_rate'].mean()
        axes[0].bar(honest_deceptive_sampling.index, honest_deceptive_sampling.values, 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0].set_title('Judge Picks Honest over Deceptive\nby Sampling Method (Precommit Only)')
        axes[0].set_ylabel('Percentage (%)')
        axes[0].set_ylim(0, 100)
        for i, v in enumerate(honest_deceptive_sampling.values):
            if not np.isnan(v):
                axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Judge picks honest over deceptive by pixel count
    if 'judge_picks_honest_over_deceptive_rate' in precommit_df.columns:
        honest_deceptive_pixel = precommit_df.groupby('pixel_count')['judge_picks_honest_over_deceptive_rate'].mean()
        axes[1].bar(honest_deceptive_pixel.index.astype(str), honest_deceptive_pixel.values, 
                   color=['orange', 'gold', 'lightpink'])
        axes[1].set_title('Judge Picks Honest over Deceptive\nby Pixel Count (Precommit Only)')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xlabel('Number of Pixels')
        axes[1].set_ylim(0, 100)
        for i, v in enumerate(honest_deceptive_pixel.values):
            if not np.isnan(v):
                axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Comparison of all metrics for precommit conditions
    metrics = ['judge_picks_true_rate', 'honest_win_rate', 'deceptive_win_rate']
    means = [precommit_df[metric].mean() for metric in metrics]
    axes[2].bar(['Judge Picks True', 'Honest Wins', 'Deceptive Wins'], means, 
               color=['gold', 'lightblue', 'lightcoral'])
    axes[2].set_title('Overall Metrics\n(Precommit Conditions)')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(means):
        axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precommit_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary(df, save_dir):
    """Create detailed statistical summary"""
    
    summary_stats = {}
    
    # Overall statistics
    summary_stats['overall'] = {
        'total_conditions': len(df),
        'total_games': df['num_games'].sum(),
        'avg_judge_picks_true': df['judge_picks_true_rate'].mean(),
        'std_judge_picks_true': df['judge_picks_true_rate'].std(),
        'avg_judge_does_not_pick_true': df['judge_does_not_pick_true_rate'].mean(),
        'std_judge_does_not_pick_true': df['judge_does_not_pick_true_rate'].std(),
        'avg_judge_picks_deceptive': df['judge_picks_deceptive_rate'].dropna().mean() if 'judge_picks_deceptive_rate' in df else None,
        'std_judge_picks_deceptive': df['judge_picks_deceptive_rate'].dropna().std() if 'judge_picks_deceptive_rate' in df else None,
        'avg_judge_picks_honest_over_deceptive': df['judge_picks_honest_over_deceptive_rate'].dropna().mean() if 'judge_picks_honest_over_deceptive_rate' in df else None,
        'std_judge_picks_honest_over_deceptive': df['judge_picks_honest_over_deceptive_rate'].dropna().std() if 'judge_picks_honest_over_deceptive_rate' in df else None,
    }
    
    # By sampling method
    summary_stats['by_sampling_method'] = {}
    for method in df['sampling_method'].unique():
        method_df = df[df['sampling_method'] == method]
        summary_stats['by_sampling_method'][method] = {
            'judge_picks_true': {
                'mean': method_df['judge_picks_true_rate'].mean(),
                'std': method_df['judge_picks_true_rate'].std()
            },
            'judge_does_not_pick_true': {
                'mean': method_df['judge_does_not_pick_true_rate'].mean(),
                'std': method_df['judge_does_not_pick_true_rate'].std()
            },
            'judge_picks_deceptive': {
                'mean': method_df['judge_picks_deceptive_rate'].dropna().mean(),
                'std': method_df['judge_picks_deceptive_rate'].dropna().std()
            },
            'judge_picks_honest_over_deceptive': {
                'mean': method_df['judge_picks_honest_over_deceptive_rate'].dropna().mean(),
                'std': method_df['judge_picks_honest_over_deceptive_rate'].dropna().std()
            }
        }
    
    # By pixel count
    summary_stats['by_pixel_count'] = {}
    for pixels in sorted(df['pixel_count'].unique()):
        pixel_df = df[df['pixel_count'] == pixels]
        summary_stats['by_pixel_count'][pixels] = {
            'judge_picks_true': {
                'mean': pixel_df['judge_picks_true_rate'].mean(),
                'std': pixel_df['judge_picks_true_rate'].std()
            },
            'judge_does_not_pick_true': {
                'mean': pixel_df['judge_does_not_pick_true_rate'].mean(),
                'std': pixel_df['judge_does_not_pick_true_rate'].std()
            },
            'judge_picks_deceptive': {
                'mean': pixel_df['judge_picks_deceptive_rate'].dropna().mean(),
                'std': pixel_df['judge_picks_deceptive_rate'].dropna().std()
            },
            'judge_picks_honest_over_deceptive': {
                'mean': pixel_df['judge_picks_honest_over_deceptive_rate'].dropna().mean(),
                'std': pixel_df['judge_picks_honest_over_deceptive_rate'].dropna().std()
            }
        }
    
    # Save statistical summary
    with open(os.path.join(save_dir, 'statistical_summary.json'), 'w') as f:
        # Convert NaN values to None for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, (np.integer, np.int64)) else k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj) if not np.isnan(obj) else None
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        json_safe_stats = convert_for_json(summary_stats)
        json.dump(json_safe_stats, f, indent=2)
    
    return summary_stats

def export_training_data(results, save_dir):
    """Export game data in format suitable for training new judge models"""
    
    training_data = {
        'metadata': {
            'source': 'mnist_debate_comprehensive_experiment',
            'timestamp': results['experiment_info']['timestamp'],
            'total_games': results['experiment_info']['total_games'],
            'conditions': list(results['conditions'].keys())
        },
        'games': []
    }
    
    game_id = 0
    for condition_name, condition_data in results['conditions'].items():
        for game in condition_data['games']:
            # Skip games that don't have complete data (e.g., test data)
            required_keys = ['true_label', 'deception_target_label', 'predicted_digit', 'move_sequence', 'revealed_pixels', 'final_probabilities']
            if not all(key in game for key in required_keys):
                continue
                
            training_sample = {
                'game_id': game_id,
                'condition': condition_name,
                'sampling_method': condition_data['sampling_method'],
                'pixel_count': condition_data['pixel_count'],
                'precommit': condition_data['precommit'],
                'true_label': game['true_label'],
                'deception_target_label': game['deception_target_label'],
                'final_predicted_digit': game['predicted_digit'],
                'move_sequence': game['move_sequence'],
                'revealed_pixels': game['revealed_pixels'],
                'final_probabilities': game['final_probabilities'],
                'game_outcome': {
                    'honest_won': game.get('honest_won', False),
                    'deceptive_won': game.get('deceptive_won', False),
                    'true_prob': game.get('true_prob', 0.0),
                    'deception_target_prob': game.get('deception_target_prob', 0.0)
                }
            }
            training_data['games'].append(training_sample)
            game_id += 1
    
    # Save training data
    with open(os.path.join(save_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Exported {len(training_data['games'])} games for potential judge training")
    return training_data

def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive MNIST debate experiment results')
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to the comprehensive_results.json file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}")
    results = load_experiment_results(args.results_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to DataFrame
    print("Converting to DataFrame for analysis...")
    df = create_analysis_dataframe(results)
    
    # Save DataFrame for future use
    df.to_csv(os.path.join(args.output_dir, 'results_dataframe.csv'), index=False)
    
    # Create main visualizations (the two graphs requested)
    print("Creating main visualizations...")
    plot_main_results(df, args.output_dir)
    
    # Create statistical summary
    print("Creating statistical summary...")
    stats = create_statistical_summary(df, args.output_dir)
    
    # Export training data
    print("Exporting training data...")
    training_data = export_training_data(results, args.output_dir)
    
    # Print summary
    print(f"\nAnalysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total conditions analyzed: {len(df)}")
    print(f"Total games: {df['num_games'].sum()}")
    print(f"Average judge picks true: {df['judge_picks_true_rate'].mean():.1f}% (±{df['judge_picks_true_rate'].std():.1f}%)")
    print(f"Average judge does not pick true: {df['judge_does_not_pick_true_rate'].mean():.1f}% (±{df['judge_does_not_pick_true_rate'].std():.1f}%)")
    if 'judge_picks_deceptive_rate' in df.columns and not df['judge_picks_deceptive_rate'].dropna().empty:
        print(f"Average judge picks deceptive (precommit): {df['judge_picks_deceptive_rate'].dropna().mean():.1f}% (±{df['judge_picks_deceptive_rate'].dropna().std():.1f}%)")
        print(f"Average judge picks honest over deceptive (precommit): {df['judge_picks_honest_over_deceptive_rate'].dropna().mean():.1f}% (±{df['judge_picks_honest_over_deceptive_rate'].dropna().std():.1f}%)")

if __name__ == '__main__':
    main() 