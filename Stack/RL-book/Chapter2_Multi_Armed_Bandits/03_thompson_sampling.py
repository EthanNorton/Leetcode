"""
Problem: Thompson Sampling for Bernoulli Bandits

Difficulty: Hard

Description:
Implement Thompson Sampling for a k-armed bandit problem with Bernoulli rewards.
The algorithm should maintain Beta distributions for each arm and use random
sampling to make decisions.

The agent should:
1. Maintain Beta distribution parameters for each arm
2. Sample from Beta distributions for action selection
3. Update distribution parameters based on observed rewards
4. Track and visualize uncertainty over time

Example:
    agent = ThompsonSamplingAgent(k=10)
    for t in range(1000):
        action = agent.select_action()
        reward = environment.step(action)  # Binary reward (0 or 1)
        agent.update(action, reward)

Learning Objectives:
1. Understand Bayesian approach to exploration
2. Implement probability matching
3. Work with Beta distributions
4. Visualize uncertainty in estimates

Hints:
1. Use Beta(1,1) as initial distribution
2. Remember Beta is conjugate prior for Bernoulli
3. Consider numerical stability in updates
"""

import numpy as np
from scipy.stats import beta

class ThompsonSamplingAgent:
    def __init__(self, k=10):
        """
        Initialize Thompson Sampling agent
        
        Args:
            k (int): Number of actions
        """
        # TODO: Initialize Beta distribution parameters
        # Use Beta(1,1) as prior for each arm
        pass
        
    def select_action(self):
        """
        Select action by sampling from Beta distributions
        
        Returns:
            int: Selected action
        """
        # TODO: Sample from Beta distribution for each arm
        # Return arm with highest sample
        pass
        
    def update(self, action, reward):
        """
        Update Beta distribution parameters
        
        Args:
            action (int): The action taken
            reward (float): The reward received (0 or 1)
        """
        # TODO: Update success and failure counts
        # Update Beta distribution parameters
        pass
        
    def get_uncertainty(self, action):
        """
        Calculate uncertainty for an action
        
        Args:
            action (int): The action to evaluate
            
        Returns:
            float: Uncertainty measure (e.g., Beta distribution variance)
        """
        # TODO: Calculate and return uncertainty measure
        pass

def test_thompson_sampling():
    # Test initialization
    agent = ThompsonSamplingAgent(k=3)
    
    # Test action selection
    actions = [agent.select_action() for _ in range(100)]
    assert len(set(actions)) > 1, "Should explore different actions"
    
    # Test updates
    agent.update(0, 1)  # Success
    agent.update(0, 0)  # Failure
    uncertainty1 = agent.get_uncertainty(0)
    
    agent.update(1, 1)  # Single update
    uncertainty2 = agent.get_uncertainty(1)
    
    assert uncertainty1 < uncertainty2, "More observations should reduce uncertainty"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_thompson_sampling()
