"""
Problem: Non-stationary Bandit with Dynamic Reward Distributions

Difficulty: Medium

Description:
Implement a non-stationary k-armed bandit where the reward distributions change over time,
and create an agent that can adapt to these changes using constant step-size parameter α.

The environment should:
1. Have k arms with initially random reward distributions
2. Change the mean rewards gradually over time using random walks
3. Include seasonal variations in rewards (optional challenge)

The agent should:
1. Use weighted average updates with constant α
2. Implement optimistic initial values for better exploration
3. Track and adapt to changing reward distributions

Example:
    bandit = NonStationaryBandit(k=10)
    agent = AdaptiveAgent(k=10, alpha=0.1, initial_value=5.0)
    
    # Run for 2000 steps
    for t in range(2000):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        bandit.random_walk()  # Update reward distributions

Learning Objectives:
1. Understand the challenges of non-stationary environments
2. Implement constant step-size updates
3. Use optimistic initial values for exploration
4. Track performance in changing environments

Hints:
1. Use small random steps for the random walk
2. Consider using a sliding window for performance tracking
3. Experiment with different α values
"""

import numpy as np

class NonStationaryBandit:
    def __init__(self, k=10, walk_std=0.01):
        """
        Initialize non-stationary k-armed bandit
        
        Args:
            k (int): Number of arms
            walk_std (float): Standard deviation for random walk steps
        """
        # TODO: Initialize mean rewards and walk parameters
        pass
        
    def pull(self, action):
        """
        Pull an arm and get reward
        
        Args:
            action (int): Which arm to pull (0 to k-1)
            
        Returns:
            float: Reward from the selected arm
        """
        # TODO: Generate reward from current distribution
        pass
        
    def random_walk(self):
        """
        Update reward distributions using random walk
        """
        # TODO: Implement random walk for mean rewards
        pass

class AdaptiveAgent:
    def __init__(self, k=10, alpha=0.1, epsilon=0.1, initial_value=5.0):
        """
        Initialize adaptive agent
        
        Args:
            k (int): Number of arms
            alpha (float): Step size parameter
            epsilon (float): Exploration rate
            initial_value (float): Optimistic initial values
        """
        # TODO: Initialize estimates with optimistic values
        pass
        
    def select_action(self):
        """
        Select action using ε-greedy strategy
        
        Returns:
            int: Selected action (0 to k-1)
        """
        # TODO: Implement ε-greedy selection
        pass
        
    def update(self, action, reward):
        """
        Update value estimates using constant step size
        
        Args:
            action (int): The action taken
            reward (float): The reward received
        """
        # TODO: Implement constant step-size update
        # Q(A) ← Q(A) + α[R - Q(A)]
        pass

class PerformanceTracker:
    def __init__(self, window_size=100):
        """
        Track agent performance
        
        Args:
            window_size (int): Size of sliding window for averaging
        """
        # TODO: Initialize tracking variables
        pass
        
    def update(self, reward):
        """
        Update performance metrics
        
        Args:
            reward (float): Latest reward
        """
        # TODO: Update sliding window average
        pass
        
    def get_performance(self):
        """
        Get current performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # TODO: Return performance statistics
        pass

# Example test cases
def test_adaptive_agent():
    np.random.seed(42)
    bandit = NonStationaryBandit(k=10)
    agent = AdaptiveAgent(k=10, alpha=0.1)
    tracker = PerformanceTracker()
    
    # Run for multiple episodes
    rewards = []
    for t in range(2000):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        tracker.update(reward)
        rewards.append(reward)
        bandit.random_walk()
    
    # Verify adaptation
    early_rewards = np.mean(rewards[:100])
    late_rewards = np.mean(rewards[-100:])
    assert abs(early_rewards - late_rewards) < 1.0, "Agent should adapt to changes"
    print("All tests passed!")

if __name__ == "__main__":
    test_adaptive_agent()
