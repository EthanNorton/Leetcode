"""
Problem: Gradient Bandit with Baseline

Difficulty: Medium

Description:
Implement a gradient bandit algorithm that learns a preference for each action
and uses a softmax distribution for action selection. Include a baseline term
to reduce variance in updates.

The agent should:
1. Maintain numerical preferences for each action
2. Use softmax action selection
3. Implement stochastic gradient ascent updates
4. Track and use a baseline (average reward) for updates

Example:
    agent = GradientBanditAgent(k=10, alpha=0.1)
    for t in range(1000):
        action = agent.select_action()
        reward = environment.step(action)
        agent.update(action, reward)

Learning Objectives:
1. Understand policy gradient methods
2. Implement softmax action selection
3. Use baselines for variance reduction
4. Apply stochastic gradient ascent

Hints:
1. Use np.exp() for numerical stability in softmax
2. Keep track of average reward for baseline
3. Update all action preferences after each step
"""

import numpy as np

class GradientBanditAgent:
    def __init__(self, k=10, alpha=0.1):
        """
        Initialize gradient bandit agent
        
        Args:
            k (int): Number of actions
            alpha (float): Step size parameter
        """
        # TODO: Initialize action preferences
        # Initialize baseline (average reward)
        pass
        
    def softmax(self):
        """
        Calculate softmax probabilities
        
        Returns:
            array: Action probabilities
        """
        # TODO: Implement numerically stable softmax
        pass
        
    def select_action(self):
        """
        Select action using softmax probabilities
        
        Returns:
            int: Selected action
        """
        # TODO: Implement softmax action selection
        pass
        
    def update(self, action, reward):
        """
        Update action preferences and baseline
        
        Args:
            action (int): The action taken
            reward (float): The reward received
        """
        # TODO: Update baseline
        # Update action preferences using gradient ascent
        # Remember to update all action preferences
        pass

def test_gradient_bandit():
    # Test initialization
    agent = GradientBanditAgent(k=3)
    
    # Test softmax probabilities
    probs = agent.softmax()
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    
    # Test preference updates
    initial_prefs = agent.preferences.copy()
    agent.update(0, 1.0)
    assert not np.array_equal(agent.preferences, initial_prefs)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gradient_bandit()
