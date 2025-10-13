"""
Problem: Simple Agent-Environment Interaction

Difficulty: Easy

Description:
Implement a basic agent that interacts with a simple environment. The environment
has a hidden optimal action, and the agent needs to learn to select this action
through trial and error.

The environment should:
1. Have a fixed set of possible actions (0 to k-1)
2. Return higher rewards for actions closer to optimal
3. Include some randomness in rewards

The agent should:
1. Keep track of average rewards for each action
2. Implement random exploration
3. Select actions based on past performance

Example:
    env = SimpleEnvironment(k=5, optimal_action=2)
    agent = SimpleAgent(k=5, exploration_rate=0.2)
    
    for t in range(100):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)

Learning Objectives:
1. Understand basic RL concepts
2. Implement simple reward tracking
3. Balance exploration and exploitation

Hints:
1. Start with uniform random exploration
2. Use arrays to track reward history
3. Consider using a simple averaging method
"""

import numpy as np

class SimpleEnvironment:
    def __init__(self, k=5, optimal_action=None):
        """
        Initialize environment
        
        Args:
            k (int): Number of possible actions
            optimal_action (int): Best action (random if None)
        """
        self.k = k

        if optimal_action is None:
            self.optimal_action = np.random.randint(k)
        else: 
            self.optimal_action = optimal_action

        
    def step(self, action):
        """
        Take action and get reward
        
        Args:
            action (int): Action to take
            
        Returns:
            float: Reward value
        """
        # Calculate reward based on distance from optimal action
        if action == self.optimal_action:
            reward = 1.0  # Best reward for optimal action
        else:
            # Less reward the further you are from optimal
            distance = abs(action - self.optimal_action)
            reward = 0.5 / (1 + distance)
        
        # Add some randomness to make it interesting
        noise = np.random.normal(0, 0.1)
        return reward + noise

class SimpleAgent:
    def __init__(self, k=5, exploration_rate=0.2):
        """
        Initialize agent
        
        Args:
            k (int): Number of possible actions
            exploration_rate (float): Probability of random action
        """
        self.k = k
        self.exploration_rate = exploration_rate
        self.reward_sums = np.zeros(k)
        self.action_counts = np.zeros(k)
        
    def choose_action(self):
        """
        Select action using simple strategy
        
        Returns:
            int: Selected action
        """
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.k)
        else:
            averages = self.reward_sums / (self.action_counts + 1e-5)
            return np.argmax(averages)
        
    def learn(self, action, reward):
        """
        Update knowledge based on reward
        
        Args:
            action (int): Action taken
            reward (float): Reward received
        """
        self.reward_sums[action] += reward
        self.action_counts[action] += 1

def test_agent_environment():
    # Test environment
    env = SimpleEnvironment(k=3, optimal_action=1)
    rewards = [env.step(1) for _ in range(10)]
    assert np.mean(rewards) > 0, "Optimal action should give positive rewards"
    
    # Test agent
    agent = SimpleAgent(k=3, exploration_rate=0.2)
    
    # Test learning
    for _ in range(50):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)
    
    # Agent should learn to prefer better actions
    action_counts = agent.action_counts
    assert np.sum(action_counts) == 50, "Should track all actions"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_agent_environment()
