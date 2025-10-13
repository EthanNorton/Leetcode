"""
Problem: Upper Confidence Bound (UCB) Action Selection

Difficulty: Easy

Description:
Implement the UCB1 algorithm for action selection in a k-armed bandit problem.
UCB1 uses the principle of "optimism in the face of uncertainty" by selecting
actions based on their potential maximum value.

The agent should:
1. Track both value estimates and number of selections for each action
2. Calculate UCB values using the formula: UCB = Q(a) + c * sqrt(ln(t)/N(a))
3. Handle initial exploration of all actions
4. Select actions that maximize the UCB value

Example:
    agent = UCBAgent(k=10, c=2.0)
    for t in range(1000):
        action = agent.select_action(t)
        reward = environment.step(action)
        agent.update(action, reward)

Learning Objectives:
1. Understand exploration through optimism
2. Implement confidence bound calculations
3. Handle the exploration-exploitation trade-off systematically

Hints:
1. Initialize action counts to avoid division by zero
2. Use numpy for efficient vector operations
3. Consider edge cases when t is small
"""

import numpy as np

class UCBAgent:
    def __init__(self, k=10, c=2.0):
        """
        Initialize UCB agent
        
        Args:
            k (int): Number of actions
            c (float): Exploration parameter
        """
        # TODO: Initialize value estimates, action counts
        pass
        
    def select_action(self, t):
        """
        Select action using UCB1 formula
        
        Args:
            t (int): Current time step
            
        Returns:
            int: Selected action
        """
        # TODO: Implement UCB1 action selection
        # Handle initial exploration
        # Calculate UCB values for each action
        # Return action with maximum UCB value
        pass
        
    def update(self, action, reward):
        """
        Update value estimates and action counts
        
        Args:
            action (int): The action taken
            reward (float): The reward received
        """
        # TODO: Update action value estimates
        # Update action selection counts
        pass

def test_ucb_agent():
    # Test initialization
    agent = UCBAgent(k=4)
    
    # Test initial exploration
    actions = [agent.select_action(t) for t in range(4)]
    assert len(set(actions)) == 4, "Should explore all actions initially"
    
    # Test UCB calculation
    agent.update(0, 1.0)  # Update first action
    action = agent.select_action(5)
    assert action != 0, "Should explore less-visited actions"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_ucb_agent()
