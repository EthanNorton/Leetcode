"""
Problem: Simple Multi-armed Bandit

Difficulty: Easy

Description:
Implement a simple k-armed bandit environment and an agent that uses the ε-greedy strategy.

The environment should:
1. Have k arms (slot machines)
2. Each arm should return rewards from a normal distribution with mean μ and standard deviation σ
3. The mean rewards for each arm should be randomly initialized

The agent should:
1. Use ε-greedy strategy for action selection
2. Keep track of value estimates for each action
3. Update estimates using incremental implementation

Example:
    bandit = KArmedBandit(k=10)  # Create a 10-armed bandit
    agent = EpsilonGreedyAgent(k=10, epsilon=0.1)
    
    # Run for 1000 steps
    for t in range(1000):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)

Learning Objectives:
1. Understand the exploration-exploitation tradeoff
2. Implement basic reward estimation
3. Practice incremental update rules

Hints:
1. Use numpy.random.normal() for reward generation
2. Remember to initialize value estimates to zero
3. Use numpy.random.random() for ε-greedy selection
"""

import numpy as np

class KArmedBandit:
    def __init__(self, k=10):
        """
        Initialize k-armed bandit
        
        Args:
            k (int): Number of arms
        """
        # TODO: Initialize the true mean rewards for each arm
        # Use np.random.normal() to generate random means
        pass
        
    def pull(self, action):
        """
        Pull an arm and get reward
        
        Args:
            action (int): Which arm to pull (0 to k-1)
            
        Returns:
            float: Reward from the selected arm
        """
        # TODO: Return a reward from normal distribution
        # Use the true mean for the selected action
        pass

class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1):
        """
        Initialize ε-greedy agent
        
        Args:
            k (int): Number of arms
            epsilon (float): Exploration rate
        """
        # TODO: Initialize value estimates and action counts
        pass
        
    def select_action(self):
        """
        Select action using ε-greedy strategy
        
        Returns:
            int: Selected action (0 to k-1)
        """
        # TODO: Implement ε-greedy action selection
        # With probability ε, choose random action
        # Otherwise choose greedy action
        pass
        
    def update(self, action, reward):
        """
        Update value estimates
        
        Args:
            action (int): The action taken
            reward (float): The reward received
        """
        # TODO: Implement incremental update rule
        # Q(A) ← Q(A) + α[R - Q(A)] where α = 1/N(A)
        pass

# Example test cases
def test_bandit():
    np.random.seed(42)  # For reproducibility
    bandit = KArmedBandit(k=10)
    agent = EpsilonGreedyAgent(k=10, epsilon=0.1)
    
    rewards = []
    for _ in range(1000):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards.append(reward)
    
    # Check average reward
    assert np.mean(rewards) > 0, "Average reward should be positive"
    print("All tests passed!")

if __name__ == "__main__":
    test_bandit()
