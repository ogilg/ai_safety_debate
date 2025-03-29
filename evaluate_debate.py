import numpy as np
import os
from tqdm import tqdm
from data_utils import get_mnist_test_dataset
from visualize_debate import visualize_debate_sequence
import random
# Import here to avoid circular imports
from debate_agents import MNISTDebateGame, Player

def simulate_debate(judge_model, image, true_label, total_pixels=None, num_simulations=100, 
                   temperature=0.5, verbose=True, sampling_mode='random', precommit=False, visualize=False,
                   save_dir=None):
    """
    Simulate a debate game between honest and deceptive agents.
    
    Args:
        judge_model: Trained MNIST judge model
        image: Full MNIST image (28x28)
        true_label: The correct digit
        total_pixels: Total number of pixels to reveal (should match judge training)
        num_simulations: Number of MCTS simulations per move
        precommit: Whether the deceptive player has to precommit to a digit
        temperature: Temperature for action selection
        verbose: Whether to print progress
        sampling_mode: How to sample pixels: 'random' or 'nonzero'
        visualize: Whether to visualize the debate game
        save_dir: Directory to save visualizations (if None, will display instead)
    
    Returns:
        Dictionary with game results
    """
    
    # Initialize the game with total_pixels and sampling_mode
    game = MNISTDebateGame(
        judge_model, 
        image, 
        true_label, 
        total_pixels=total_pixels,
        sampling_mode=sampling_mode
    )
    
    # Get max_turns from the game
    max_turns = game.max_turns
    
    # Initialize agents
    honest_player = Player(judge_model, max_turns, num_simulations=num_simulations, player_type='honest', precommit=precommit)
    deceptive_player = Player(judge_model, max_turns, num_simulations=num_simulations, player_type='deceptive', precommit=precommit)
    
    # Play the game
    while not game.is_terminal():
        # Get current state
        state = game.get_state()
        
        # Determine current player
        current_player = honest_player if state.current_player == 0 else deceptive_player
        
        # Get action from player with temperature
        action = current_player.decide_move(state, temperature=temperature)
        
        # Apply action
        game.apply_action(action)
        
        # Optional: print progress
        if verbose:
            player_name = 'Honest' if state.current_player == 0 else 'Deceptive'
            print(f"Turn {len(game.revealed_pixels)}: {player_name} player revealed pixel at {action}")
    
    # Get final result
    result = game.get_game_result()
    
    # Add the full image to the result
    result['full_image'] = image.tolist() if isinstance(image, np.ndarray) else image
    
    # Print summary
    if verbose:
        print(f"\nGame finished after {len(game.revealed_pixels)} turns ({game.total_pixels} total pixels)")
        print(f"True digit: {result['true_label']}, Deception Target digit: {result['deception_target_label']}")
        print(f"Judge prediction: {result['predicted_digit']}")
        print(f"Probability for true digit: {result['true_prob']:.4f}")
        print(f"Probability for deception target digit: {result['deception_target_prob']:.4f}")
        print(f"Winner: {'Honest' if result['honest_won'] else 'Deceptive' if result['deceptive_won'] else 'Neither'}")
    
    # Visualize the game if requested
    if visualize:
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Define save paths
            sequence_path = os.path.join(save_dir, f'game_sequence.png')
        else:
            sequence_path = None
        
        # Visualize sequence
        visualize_debate_sequence(result, judge_model, save_path=sequence_path)
    
    return result

def evaluate_debate_performance(judge_model, test_dataset=None, num_games=100, total_pixels=None, 
                               num_simulations=100, precommit=False, temperature=0.5, sampling_mode='random',
                               visualize=False, save_dir=None):
    """
    Evaluate the performance of the debate players across multiple games using vectorized operations.
    
    Args:
        judge_model: Trained MNIST judge model
        test_dataset: MNIST test dataset (if None, will be loaded)
        num_games: Number of games to simulate
        total_pixels: Total number of pixels to reveal (if None, use judge model value)
        num_simulations: Number of MCTS simulations per move
        precommit: Whether the deceptive player has to precommit to a digit
        temperature: Temperature for action selection
        sampling_mode: How to sample pixels ('random' or 'nonzero')
        visualize: Whether to visualize debate games
        save_dir: Directory to save results and visualizations
        
    Returns:
        results: Dictionary containing evaluation results
    """
    # Load test dataset if not provided
    if test_dataset is None:
        test_dataset = get_mnist_test_dataset()
    
    results = {
        'honest_wins': 0,
        'deceptive_wins': 0,
        'draws': 0,
        'true_prob_higher': 0,
        'deception_target_prob_higher': 0,
        'games': []
    }
    
    # Sample random images from test dataset using numpy
    indices = np.random.choice(len(test_dataset), num_games, replace=False)
    
    # Create save directory if needed for visualizations
    if visualize and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine which games to visualize (randomly select 1%)
        num_to_visualize = max(1, int(num_games * 0.01))  # At least 1 game
        games_to_visualize = set(random.sample(range(num_games), num_to_visualize))
    else:
        games_to_visualize = set()
    
    # Use tqdm for progress tracking
    for i, idx in enumerate(tqdm(indices, desc="Simulating debates")):
        # Get image and label
        image, true_label = test_dataset[idx]
        image_np = image.squeeze().numpy()  # Convert to numpy array
        
        # Determine if this game should be visualized
        should_visualize = visualize and (i in games_to_visualize)
        
        # Create game-specific save directory if visualizing this game
        game_save_dir = None
        if should_visualize and save_dir:
            game_save_dir = os.path.join(save_dir, f'game_{i+1}')
            os.makedirs(game_save_dir, exist_ok=True)
        
        # Simulate debate with total_pixels and sampling_mode
        game_result = simulate_debate(
            judge_model, 
            image_np, 
            int(true_label),
            total_pixels=total_pixels,
            num_simulations=num_simulations,
            temperature=temperature,
            verbose=False,
            sampling_mode=sampling_mode,
            precommit=precommit,
            visualize=should_visualize,
            save_dir=game_save_dir
        )
        
        # Update statistics
        results['honest_wins'] += game_result['honest_won']
        results['deceptive_wins'] += game_result['deceptive_won']
        results['draws'] += (not game_result['honest_won'] and not game_result['deceptive_won'])
        results['true_prob_higher'] += (game_result['true_prob'] > game_result['deception_target_prob'])
        results['deception_target_prob_higher'] += (game_result['true_prob'] <= game_result['deception_target_prob'])
            
        # Store detailed game result
        results['games'].append(game_result)
    
    # Calculate percentages using vectorized operations
    total_games = len(results['games'])
    results['honest_win_rate'] = results['honest_wins'] / total_games * 100
    results['deceptive_win_rate'] = results['deceptive_wins'] / total_games * 100
    results['draw_rate'] = results['draws'] / total_games * 100
    results['true_prob_higher_rate'] = results['true_prob_higher'] / total_games * 100
    results['deception_target_prob_higher_rate'] = results['deception_target_prob_higher'] / total_games * 100
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Total games: {total_games}")
    print(f"Honest player win rate: {results['honest_win_rate']:.2f}%")
    print(f"Deceptive player win rate: {results['deceptive_win_rate']:.2f}%")
    print(f"Draw rate: {results['draw_rate']:.2f}%")
    print(f"True probability higher rate: {results['true_prob_higher_rate']:.2f}%")
    print(f"Deception target probability higher rate: {results['deception_target_prob_higher_rate']:.2f}%")
    
    if visualize:
        print(f"Visualized {len(games_to_visualize)} games (10% of total)")
    
    return results