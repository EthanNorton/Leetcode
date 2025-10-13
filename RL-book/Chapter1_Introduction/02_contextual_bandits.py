"""
Problem: Contextual Bandits with Weather States

Difficulty: Medium

Description:
Implement a contextual bandit problem where the optimal action depends on a
context (weather state). The agent must learn which actions work best in
different weather conditions.

The environment should:
1. Have different weather states (sunny, rainy, cloudy)
2. Different optimal actions for each weather state
3. Provide context (weather) before each decision
4. Include realistic reward structures

The agent should:
1. Learn separate value estimates for each context
2. Implement context-specific exploration
3. Track performance in different conditions

Example:
    env = WeatherBandit(n_actions=3)
    agent = ContextualAgent(n_contexts=3, n_actions=3)
    
    for t in range(1000):
        context = env.get_context()
        action = agent.choose_action(context)
        reward = env.step(action)
        agent.learn(context, action, reward)

Learning Objectives:
1. Understand contextual decision making
2. Implement state-dependent policies
3. Handle multiple learning scenarios
4. Track context-specific performance

Hints:
1. Use dictionaries or 2D arrays for context-action values
2. Consider weather transitions
3. Implement context-specific exploration rates
"""

import numpy as np

class WeatherBandit:
    def __init__(self, n_actions=3):
        """
        Initialize weather-based bandit
        
        Args:
            n_actions (int): Number of possible actions
        """
        # TODO: Initialize environment
        # Set up weather states
        # Define optimal actions for each weather
        # Set up transition probabilities
        pass
        
    def get_context(self):
        """
        Get current weather state
        
        Returns:
            int: Current weather state
        """
        # TODO: Return current weather
        # Update weather if needed
        pass
        
    def step(self, action):
        """
        Take action and get reward
        
        Args:
            action (int): Action to take
            
        Returns:
            float: Reward value
        """
        # TODO: Calculate reward based on weather and action
        # Include randomness
        pass

class ContextualAgent:
    def __init__(self, n_contexts=3, n_actions=3, base_exploration=0.1):
        """
        Initialize contextual agent
        
        Args:
            n_contexts (int): Number of contexts (weather states)
            n_actions (int): Number of possible actions
            base_exploration (float): Base exploration rate
        """
        # TODO: Initialize agent parameters
        # Set up context-action value estimates
        # Initialize exploration parameters
        pass
        
    def choose_action(self, context):
        """
        Select action based on context
        
        Args:
            context (int): Current context (weather)
            
        Returns:
            int: Selected action
        """
        # TODO: Implement context-based action selection
        # Use context-specific exploration
        pass
        
    def learn(self, context, action, reward):
        """
        Update knowledge for specific context
        
        Args:
            context (int): The context
            action (int): Action taken
            reward (float): Reward received
        """
        # TODO: Update value estimates for context-action pair
        # Update context-specific counters
        pass
        
    def get_policy(self):
        """
        Get current policy for each context
        
        Returns:
            dict: Mapping of context to best actions
        """
        # TODO: Return best actions for each context
        pass

def test_contextual_bandit():
    # Test environment
    env = WeatherBandit(n_actions=3)
    context = env.get_context()
    assert 0 <= context < 3, "Invalid context"
    
    # Test agent
    agent = ContextualAgent(n_contexts=3, n_actions=3)
    
    # Test learning
    rewards = []
    for _ in range(100):
        context = env.get_context()
        action = agent.choose_action(context)
        reward = env.step(action)
        agent.learn(context, action, reward)
        rewards.append(reward)
    
    # Check if agent learns context-specific policies
    policy = agent.get_policy()
    assert len(policy) == 3, "Should have policy for each context"
    
    # Check if average reward improves
    early_rewards = np.mean(rewards[:20])
    late_rewards = np.mean(rewards[-20:])
    assert late_rewards >= early_rewards, "Agent should improve over time"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_contextual_bandit()
