"""
Problem: Finding Optimal Policy in a Stochastic Environment

Difficulty: Hard

Description:
Implement a stochastic environment with multiple goals and obstacles, then
find the optimal policy using value iteration. The environment should include
wind effects that make actions probabilistic.

The environment should:
1. Have multiple goal states with different rewards
2. Include obstacles and penalties
3. Implement stochastic transitions (wind effect)
4. Support variable starting positions

Tasks:
1. Implement stochastic grid environment
2. Calculate optimal value function
3. Extract optimal policy
4. Visualize policy and value function

Example:
    env = StochasticGrid()
    optimal_value = find_optimal_value_function(env)
    optimal_policy = extract_policy(optimal_value, env)
    visualize_policy(optimal_policy)

Learning Objectives:
1. Understand optimal policies
2. Handle stochastic transitions
3. Implement value iteration
4. Extract policies from value functions

Hints:
1. Use probability matrices for transitions
2. Consider vectorized operations for speed
3. Implement efficient policy extraction
"""

import numpy as np

class StochasticGrid:
    def __init__(self, size=5, wind_prob=0.2):
        """
        Initialize stochastic grid environment
        
        Args:
            size (int): Grid size
            wind_prob (float): Probability of wind effect
        """
        # TODO: Initialize grid
        # Set up goals and obstacles
        # Define transition probabilities
        pass
        
    def get_transition_probs(self, state, action):
        """
        Get transition probabilities for state-action pair
        
        Args:
            state (tuple): Current state
            action (int): Action to take
            
        Returns:
            dict: Mapping of next states to probabilities
        """
        # TODO: Calculate transition probabilities
        # Include wind effects
        # Handle boundaries and obstacles
        pass
        
    def get_reward(self, state, action, next_state):
        """
        Get reward for transition
        
        Args:
            state (tuple): Current state
            action (int): Action taken
            next_state (tuple): Resulting state
            
        Returns:
            float: Reward value
        """
        # TODO: Calculate reward
        # Consider goals and penalties
        pass

def find_optimal_value_function(env, gamma=0.9, theta=0.001):
    """
    Find optimal value function using value iteration
    
    Args:
        env (StochasticGrid): Environment
        gamma (float): Discount factor
        theta (float): Convergence threshold
        
    Returns:
        array: Optimal value function
    """
    # TODO: Implement value iteration
    # Initialize value function
    # Iterate until convergence
    # Handle stochastic transitions
    pass

def extract_policy(value_function, env, gamma=0.9):
    """
    Extract optimal policy from value function
    
    Args:
        value_function (array): Optimal value function
        env (StochasticGrid): Environment
        gamma (float): Discount factor
        
    Returns:
        array: Optimal policy
    """
    # TODO: Extract optimal policy
    # For each state, find best action
    # Consider stochastic transitions
    pass

def test_stochastic_grid():
    # Test environment
    env = StochasticGrid(size=3)
    
    # Test transition probabilities
    state = (0, 0)
    action = 0  # up
    probs = env.get_transition_probs(state, action)
    assert sum(probs.values()) == 1.0, "Probabilities should sum to 1"
    
    # Test value iteration
    value_function = find_optimal_value_function(env)
    assert value_function.shape == (3, 3), "Incorrect value function shape"
    
    # Test policy extraction
    policy = extract_policy(value_function, env)
    assert policy.shape == (3, 3), "Incorrect policy shape"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_stochastic_grid()
