"""
Problem: Action-Value Function for Tic-Tac-Toe

Difficulty: Medium

Description:
Implement a Tic-Tac-Toe environment and calculate action-value functions
for a given policy. This will help understand how state-action pairs are
evaluated in a practical gaming scenario.

The environment should:
1. Implement complete Tic-Tac-Toe mechanics
2. Track game state and valid moves
3. Provide rewards (+1 win, -1 loss, 0 draw)
4. Support two players (agent vs random opponent)

Tasks:
1. Implement Tic-Tac-Toe environment
2. Calculate action-value function for a given policy
3. Implement policy evaluation
4. Visualize action values for different board states

Example:
    env = TicTacToe()
    policy = RandomPolicy()
    q_function = calculate_action_values(env, policy)
    best_action = select_action(q_function, current_state)

Learning Objectives:
1. Understand action-value functions
2. Implement game mechanics as an MDP
3. Handle large state spaces
4. Work with stochastic opponents

Hints:
1. Use efficient state representation
2. Consider symmetries in the game
3. Use sparse data structures for large state spaces
"""

import numpy as np

class TicTacToe:
    def __init__(self):
        """
        Initialize Tic-Tac-Toe environment
        """
        # TODO: Initialize game board
        # Track current player
        # Define win conditions
        pass
        
    def reset(self):
        """
        Reset the game to initial state
        
        Returns:
            array: Initial state
        """
        # TODO: Reset board state
        pass
        
    def get_valid_actions(self, state):
        """
        Get valid actions for current state
        
        Args:
            state (array): Current game state
            
        Returns:
            list: Valid actions
        """
        # TODO: Return list of valid moves
        pass
        
    def step(self, action):
        """
        Take action and return new state
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # TODO: Implement game step
        # Make move
        # Check win/draw conditions
        # Handle opponent move
        pass
        
    def get_reward(self, state):
        """
        Calculate reward for current state
        
        Args:
            state (array): Current state
            
        Returns:
            float: Reward value
        """
        # TODO: Calculate reward
        # Check win/loss/draw
        pass

class RandomPolicy:
    def __init__(self):
        """
        Initialize random policy
        """
        pass
        
    def select_action(self, state, valid_actions):
        """
        Select random action from valid actions
        
        Args:
            state (array): Current state
            valid_actions (list): List of valid actions
            
        Returns:
            int: Selected action
        """
        # TODO: Implement random action selection
        pass

def calculate_action_values(env, policy, gamma=0.9, episodes=1000):
    """
    Calculate action-value function for given policy
    
    Args:
        env (TicTacToe): Environment
        policy (RandomPolicy): Policy to evaluate
        gamma (float): Discount factor
        episodes (int): Number of episodes
        
    Returns:
        dict: Action-value function
    """
    # TODO: Implement action-value calculation
    # Initialize Q-values
    # Run episodes
    # Update Q-values
    pass

def test_tictactoe():
    # Test environment
    env = TicTacToe()
    state = env.reset()
    
    # Test valid actions
    valid_actions = env.get_valid_actions(state)
    assert len(valid_actions) == 9, "Should have 9 valid moves initially"
    
    # Test game mechanics
    action = valid_actions[0]
    next_state, reward, done, _ = env.step(action)
    assert np.sum(np.abs(next_state)) > 0, "State should change after move"
    
    # Test policy
    policy = RandomPolicy()
    action = policy.select_action(state, valid_actions)
    assert action in valid_actions, "Selected action should be valid"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_tictactoe()
