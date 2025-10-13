"""
Problem: GridWorld State Value Function

Difficulty: Easy

Description:
Implement a simple GridWorld environment and calculate the state-value function
for a given policy. The GridWorld should follow the example from the textbook
with terminal states and negative rewards for each move.

The environment should:
1. Implement a 4x4 grid with terminal states
2. Support four actions: up, down, left, right
3. Give -1 reward for each move
4. Handle boundary conditions

Tasks:
1. Implement the GridWorld environment
2. Calculate state-value function for equiprobable random policy
3. Visualize the value function

Example:
    env = GridWorld(4, 4)
    value_function = calculate_value_function(env)
    visualize_value_function(value_function)

Learning Objectives:
1. Understand state-value functions
2. Implement Bellman equation
3. Work with terminal and non-terminal states
4. Handle boundary conditions in MDPs

Hints:
1. Use numpy arrays for the grid
2. Consider matrix operations for vectorized updates
3. Remember to handle terminal states separately
"""

import numpy as np

class GridWorld:
    def __init__(self, width=4, height=4):
        """
        Initialize GridWorld
        
        Args:
            width (int): Grid width
            height (int): Grid height
        """
        # TODO: Initialize grid structure
        # Define terminal states
        # Define possible actions
        pass
        
    def get_next_state(self, state, action):
        """
        Get next state given current state and action
        
        Args:
            state (tuple): Current state coordinates
            action (int): Action to take
            
        Returns:
            tuple: Next state coordinates
            float: Reward
        """
        # TODO: Implement state transition
        # Handle boundary conditions
        # Return next state and reward
        pass
        
    def is_terminal(self, state):
        """
        Check if state is terminal
        
        Args:
            state (tuple): State to check
            
        Returns:
            bool: True if terminal state
        """
        # TODO: Check if state is terminal
        pass

def calculate_value_function(env, gamma=0.9, theta=0.001):
    """
    Calculate state-value function for random policy
    
    Args:
        env (GridWorld): Environment
        gamma (float): Discount factor
        theta (float): Convergence threshold
        
    Returns:
        array: State-value function
    """
    # TODO: Implement value function calculation
    # Initialize value function
    # Iterate until convergence
    # Return final value function
    pass

def visualize_value_function(value_function):
    """
    Visualize the value function
    
    Args:
        value_function (array): State-value function to visualize
    """
    # TODO: Implement visualization
    # Create readable output format
    pass

def test_gridworld():
    # Test environment
    env = GridWorld()
    
    # Test state transitions
    state = (0, 0)
    next_state, reward = env.get_next_state(state, 0)  # Try moving up
    assert reward == -1, "Should get -1 reward for each move"
    
    # Test terminal states
    assert env.is_terminal((0, 0)) == False, "Should not be terminal"
    assert env.is_terminal((3, 3)) == True, "Should be terminal"
    
    # Test value function
    value_function = calculate_value_function(env)
    assert value_function.shape == (4, 4), "Incorrect value function shape"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gridworld()
