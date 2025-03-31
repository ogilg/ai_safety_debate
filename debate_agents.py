import torch
import numpy as np
import random
import os
from judge_model import create_sparse_image
from data_utils import get_sparse_image
import matplotlib
# Use non-interactive backend if running without display
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')

class MNISTGameState:
    def __init__(self, image, true_label, deception_target_label, revealed_pixels=None, current_player=0, sampling_mode='random'):
        # Convert image to float32 to reduce memory usage
        self.image = image  
        if not isinstance(self.image, np.ndarray):
            self.image = np.array(self.image, dtype=np.float32)
        elif self.image.dtype != np.float32:
            self.image = self.image.astype(np.float32)
            
        self.true_label = true_label
        self.deception_target_label = deception_target_label
        self.current_player = current_player
        self.sampling_mode = sampling_mode
        
        # Create a mask for faster legal action checking (using bool_ is more memory efficient)
        self.revealed_mask = np.zeros((28, 28), dtype=np.bool_)
        
        # Store revealed pixels more efficiently
        if revealed_pixels is None or len(revealed_pixels) == 0:
            self.revealed_pixels = []
            # No need to update mask as it's already all zeros
        else:
            # Store the original list format for compatibility
            self.revealed_pixels = revealed_pixels
            
            # Update the mask for fast operations
            for x, y, _ in revealed_pixels:
                self.revealed_mask[y, x] = True
    
    def get_legal_actions(self):
        """Return coordinates of pixels that haven't been revealed yet using vectorized operations"""
        if self.sampling_mode == 'random':
            # Use numpy to find unrevealed pixels (much faster than list comprehension)
            y_indices, x_indices = np.where(~self.revealed_mask)
            return list(zip(x_indices, y_indices))
        elif self.sampling_mode == 'nonzero':
            # Find unrevealed non-zero pixels
            unrevealed = ~self.revealed_mask
            nonzero = self.image > 0
            valid_pixels = np.logical_and(unrevealed, nonzero)
            y_indices, x_indices = np.where(valid_pixels)
            return list(zip(x_indices, y_indices))
    
    def apply_action(self, action):
        """Reveal a pixel and return new state"""
        x, y = action
        pixel_value = float(self.image[y, x])
        
        # Create a new list for revealed pixels
        new_revealed = self.revealed_pixels.copy()
        new_revealed.append((x, y, pixel_value))
        
        # Create a new state with updated mask
        new_state = MNISTGameState(
            self.image, 
            self.true_label,
            self.deception_target_label,
            new_revealed, 
            1 - self.current_player  # Switch player
        )
        
        return new_state
    
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        # Set prior to 1 divided by the number of nonzero pixels in the image
        self.prior = 1.0 / (np.count_nonzero(state.image) - len(state.revealed_pixels))
        
    def is_expanded(self):
        return len(self.children) > 0
        
    def select_child(self):
        """Select child using PUCT formula with 1/num_children as the exploration constant (vectorized)"""
        if not self.children:
            return None, None
        
        c_puct = 1.0 
        
        # Get all actions and children
        actions = list(self.children.keys())
        children = list(self.children.values())
        
        # Calculate Q-values (exploitation term)
        visit_counts = np.array([child.visit_count for child in children])
        value_sums = np.array([child.value_sum for child in children])
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            q_values = np.divide(value_sums, visit_counts)
        q_values = np.nan_to_num(q_values)  # Replace NaN with 0
        
        # Calculate U-values (exploration term)
        priors = np.array([child.prior for child in children])
        u_values = c_puct * priors * np.sqrt(self.visit_count) / (1 + visit_counts)
        
        # Calculate PUCT scores
        puct_scores = q_values + u_values
        
        top_indices = np.where(puct_scores == np.max(puct_scores))[0]
        # If there's a tie, randomly select one of the maximum indices
        best_idx = np.random.choice(top_indices)
        best_action = actions[best_idx]
        
        return best_action, self.children[best_action]

class Player:
    def __init__(self, judge_model, max_turns, num_simulations=100, player_type='honest', precommit=False, batch_size=8):
        self.judge_model = judge_model
        self.max_turns = max_turns
        self.num_simulations = num_simulations
        self.player_type = player_type
        self.precommit = precommit
        self.batch_size = batch_size
        
    def decide_move(self, state, temperature=0.5):
        """Run MCTS search from the given state to decide on a move"""
        root = MCTSNode(state)
        
        # Process simulations in batches for efficiency
        for sim_batch_start in range(0, self.num_simulations, self.batch_size):
            # Determine batch size for this iteration
            current_batch_size = min(self.batch_size, self.num_simulations - sim_batch_start)
            
            # Lists to store nodes and states for batch evaluation
            leaf_nodes = []
            leaf_states = []
            search_paths = []
            
            # Selection and expansion phase for each simulation in the batch
            for _ in range(current_batch_size):
                node = root
                search_path = [node]
                
                # Selection phase - traverse tree until we find a leaf
                while node.is_expanded() and len(node.children) > 0:
                    action, node = node.select_child()
                    search_path.append(node)
                
                # If we've reached a terminal state
                if len(node.state.revealed_pixels) >= 2 * self.max_turns:
                    # Evaluate terminal state immediately
                    value = self._evaluate_terminal(node.state)
                    # Backpropagate the value
                    self._backpropagate(search_path, value)
                else:
                    # Expansion phase - expand leaf if not terminal
                    if not node.is_expanded():
                        self._expand(node)
                    
                    # Store the node, state, and search path for batch evaluation
                    leaf_nodes.append(node)
                    leaf_states.append(node.state)
                    search_paths.append(search_path)
            
            # If there are leaf nodes to evaluate in batch
            if leaf_nodes:
                # Batch simulate and evaluate the leaf states
                values = self._batch_simulate_and_evaluate(leaf_states)
                
                # Backpropagate values for each simulation
                for i, value in enumerate(values):
                    self._backpropagate(search_paths[i], value)
        
        # Return action sampled based on visit counts and temperature
        return self._select_action(root, temperature)
    
    def _batch_simulate_and_evaluate(self, states):
        """Simulate and evaluate a batch of states"""
        # Random playouts to terminal states
        terminal_states = []
        
        # Perform random playouts for all states
        for state in states:
            terminal_state = self._simulate_single(state)
            terminal_states.append(terminal_state)
        
        # Batch evaluate all terminal states at once
        return self._batch_evaluate_terminal(terminal_states)
    
    def _simulate_single(self, state):
        """Simulate a single random playout with memory optimization"""
        # Create a new state object reusing the image reference 
        # to avoid duplicating the large image array
        sim_state = MNISTGameState(
            state.image,  # This is just a reference, not a copy
            state.true_label,
            state.deception_target_label,
            state.revealed_pixels.copy(),  # Need to copy to avoid modifying original state
            state.current_player,
            state.sampling_mode
        )
        
        max_total_turns = 2 * self.max_turns  # Both players combined
        current_turns = len(sim_state.revealed_pixels)
        
        # Pre-allocate an array for legal actions to reduce memory allocations
        legal_actions = sim_state.get_legal_actions()
        
        while current_turns < max_total_turns and legal_actions:
            # Choose random action
            action_idx = np.random.randint(len(legal_actions))
            action = legal_actions[action_idx]
            
            # Apply action and update state
            sim_state = sim_state.apply_action(action)
            current_turns += 1
            
            # Get new legal actions
            legal_actions = sim_state.get_legal_actions()
        
        return sim_state
    
    def _expand(self, node):
        """Expand node by adding all possible children"""
        for action in node.state.get_legal_actions():
            child_state = node.state.apply_action(action)
            child_node = MCTSNode(child_state, parent=node, action=action)
            node.children[action] = child_node
    
    def _backpropagate(self, search_path, value):
        """Update statistics for all nodes in search path"""
        for node in search_path:
            node.visit_count += 1
            
            # Value is from the perspective of the player who just moved
            # So we need to negate it for the parent (opponent's perspective)
            if node != search_path[-1]:  # Not the leaf node
                node.value_sum += (1 - value)
            else:
                node.value_sum += value
    
    def _select_action(self, root, temperature=1.0):
        """
        Select action based on visit count distribution with temperature using vectorized operations.
        
        Args:
            root: The root node
            temperature: Controls randomness (0 = deterministic, higher = more random)
        
        Returns:
            Selected action
        """
        # Get all actions and visit counts as numpy arrays
        actions = list(root.children.keys())
        visit_counts = np.array([child.visit_count for _, child in root.children.items()])
        
        if temperature == 0 or len(actions) == 1:
            # Deterministic selection (choose highest visit count)
            best_idx = np.argmax(visit_counts)
            return actions[best_idx]
        else:
            # Apply temperature to visit counts using vectorized operation
            if temperature != 1.0:
                visit_counts = np.power(visit_counts, 1.0 / temperature)
            
            # Convert to probability distribution
            total_visits = np.sum(visit_counts)
            if total_visits == 0:
                # If no visits yet, choose randomly
                return random.choice(actions)
            
            probabilities = visit_counts / total_visits
            
            # Sample action according to the probability distribution
            action_indices = np.arange(len(actions))
            chosen_idx = np.random.choice(action_indices, p=probabilities)
            return actions[chosen_idx]
    
    def _evaluate_terminal(self, state):
        """Evaluate a single terminal state (kept for compatibility)"""
        sparse_image = get_sparse_image(state.revealed_pixels)
        _, probabilities = self.judge_model.predict(sparse_image)
        return self._get_terminal_state_value(state, probabilities)
    
    def _batch_evaluate_terminal(self, states):
        """Evaluate a batch of terminal states at once"""
        # Create a batch of sparse images
        sparse_images = [get_sparse_image(state.revealed_pixels) for state in states]
        
        # Get batch predictions using the unified method
        _, probabilities = self.judge_model.batch_predict(sparse_images)
        
        # Extract relevant probabilities based on player type
        values = []
        for i, state in enumerate(states):
            state_probs = probabilities[i]
            value = self._get_terminal_state_value(state, state_probs)
            values.append(value)
        
        return values
    
    def _get_terminal_state_value(self, state, probabilities):
        """Evaluate a single terminal state"""
        if self.precommit:
            true_label_score = probabilities[state.true_label] / probabilities.max()
            if self.player_type == 'honest':
                # Honest agent wants high probability for true label relative to other labels
                return true_label_score
            else:
                # Deceptive agent wants low probability for true label
                return 1 - true_label_score
        else:
            # Score needs to be between 0 and 1
            true_v_deception_score = (1 + probabilities[state.true_label] - probabilities[state.deception_target_label]) / 2
            if self.player_type == 'honest':
                # Honest agent wants high probability for true label relative to other labels
                return true_v_deception_score
            else:
                # Deceptive agent wants low probability for true label
                return 1 - true_v_deception_score
        
class MNISTDebateGame:
    def __init__(self, judge_model, image, true_label, deception_target_label=None, total_pixels=None, sampling_mode='random'):
        """
        Initialize a debate game between honest and deceptive agents.
        
        Args:
            judge_model: Trained MNIST judge model
            image: Full MNIST image (28x28)
            true_label: The correct digit
            deception_target_label: Target incorrect digit for deceptive agent (if None, will be chosen automatically)
            total_pixels: Total number of pixels to reveal (should match judge training)
                         If None, will try to extract from judge_model
            sampling_mode: How to sample pixels: 'random' or 'nonzero'
        """
        self.judge_model = judge_model
        self.image = image
        self.true_label = true_label
        self.revealed_pixels = []
        self.sampling_mode = sampling_mode
        
        # Add a mask for faster operations
        self.revealed_mask = np.zeros((28, 28), dtype=bool)
        
        # Determine the total number of pixels to reveal
        if total_pixels is None:
            # Try to get the number of pixels from the judge model's training args
            if hasattr(judge_model, 'num_pixels'):
                self.total_pixels = judge_model.num_pixels
            else:
                # Default to 10 if not specified
                self.total_pixels = 10
                print("Warning: Could not determine judge model's training pixel count. Using default of 10.")
        else:
            self.total_pixels = total_pixels
            
        # Calculate max turns per player (each player reveals half the pixels)
        self.max_turns = self.total_pixels // 2
        
        # If total_pixels is odd, add one more turn for the honest player
        self.extra_honest_turn = (self.total_pixels % 2) == 1
        
        # If deception_target_label is not provided, choose the second most likely digit
        if deception_target_label is None:
            # Choose a random digit that's not the true label
            possible_digits = [i for i in range(10) if i != true_label]
            self.deception_target_label = random.choice(possible_digits)
            
        else:
            self.deception_target_label = deception_target_label
            
        self.current_player = 0  # 0 for honest, 1 for deceptive
        
    def get_state(self):
        """Return the current game state"""
        return MNISTGameState(
            self.image, 
            self.true_label,
            self.deception_target_label,
            self.revealed_pixels.copy(),
            self.current_player,
            self.sampling_mode
        )
    
    def apply_action(self, action):
        """Apply an action (reveal a pixel) and switch players"""
        x, y = action
        pixel_value = float(self.image[y, x])
        
        # Add to the revealed pixels list
        self.revealed_pixels.append((x, y, pixel_value))
        
        # Update the mask (faster than checking the list each time)
        self.revealed_mask[y, x] = True
        
        # Switch player
        self.current_player = 1 - self.current_player
        
    def is_terminal(self):
        """Check if the game is over using vectorized operations"""
        # Count revealed pixels using the mask (faster than len(self.revealed_pixels))
        num_revealed = np.sum(self.revealed_mask)
        
        if self.extra_honest_turn:
            # If we have an odd number of total pixels, honest player gets one extra turn
            return num_revealed >= self.total_pixels
        else:
            # Otherwise, each player gets the same number of turns
            return num_revealed >= 2 * self.max_turns
    
    def get_game_result(self):
        """Get the final result of the game"""
        # Use sparse_image directly with the revealed_pixels list
        sparse_image = create_sparse_image(self.revealed_pixels)
        predicted_digit, probabilities = self.judge_model.predict(sparse_image)
        
        # Rest of the function unchanged
        honest_won = (predicted_digit == self.true_label)
        deceptive_won = (predicted_digit == self.deception_target_label)
        
        true_prob = probabilities[self.true_label]
        deception_target_prob = probabilities[self.deception_target_label]
        
        return {
            'predicted_digit': predicted_digit,
            'probabilities': probabilities,
            'honest_won': honest_won,
            'deceptive_won': deceptive_won,
            'true_prob': true_prob,
            'deception_target_prob': deception_target_prob,
            'true_label': self.true_label,
            'deception_target_label': self.deception_target_label,
            'revealed_pixels': self.revealed_pixels
        } 