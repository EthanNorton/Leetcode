"""
Problem: Implement Epsilon-Greedy Action Selection

Difficulty: Easy-Medium
Category: Exploration vs Exploitation

Description:
Implement the epsilon-greedy strategy:
- With probability epsilon: choose random action (explore)
- With probability 1-epsilon: choose best action (exploit)

This is THE MOST COMMON exploration strategy in RL!

Example:
    selector = EpsilonGreedy(n_actions=3, epsilon=0.2)
    selector.set_values([0.5, 0.8, 0.3])  # Action 1 has highest value
    
    # Run 100 times - should pick action 1 about 80% of time
    actions = [selector.choose() for _ in range(100)]

Learning Objectives:
1. Use random numbers for decision making
2. Implement explore/exploit trade-off
3. Find maximum values in arrays
"""

import numpy as np

class EpsilonGreedy:
    def __init__(self, n_actions=3, epsilon=0.1):
        """
        Initialize epsilon-greedy selector
        
        Args:
            n_actions (int): Number of possible actions
            epsilon (float): Exploration rate (0.1 = 10% random)
            
        TODO: Store the parameters
        TODO: Initialize value estimates to zeros
        """
        # YOUR CODE HERE
        pass
        
    def set_values(self, values):
        """
        Set the current value estimates
        
        Args:
            values (array or list): Value estimate for each action
            
        TODO: Store these values
        HINT: Convert to numpy array: np.array(values)
        """
        # YOUR CODE HERE
        pass
        
    def choose(self):
        """
        Choose an action using epsilon-greedy strategy
        
        Returns:
            int: Selected action
            
        TODO: 
        1. Generate random number between 0 and 1
        2. If random < epsilon: return random action
        3. Else: return action with highest value
        
        HINT: np.random.random() gives random number 0-1
        HINT: np.random.randint(self.n_actions) gives random action
        HINT: np.argmax(self.values) gives action with max value
        """
        # YOUR CODE HERE
        pass


# ===========================================================================
# TESTS
# ===========================================================================

def test_epsilon_greedy():
    print("Testing Epsilon-Greedy...\n")
    
    # Test 1: Always exploit (epsilon=0)
    print("Test 1: Pure exploitation (epsilon=0)")
    selector = EpsilonGreedy(n_actions=3, epsilon=0.0)
    selector.set_values([0.3, 0.8, 0.5])
    
    actions = [selector.choose() for _ in range(20)]
    unique_actions = set(actions)
    assert unique_actions == {1}, f"❌ With epsilon=0, should always pick action 1, got {unique_actions}"
    print("✓ Always picks best action when epsilon=0\n")
    
    # Test 2: Always explore (epsilon=1)
    print("Test 2: Pure exploration (epsilon=1)")
    selector = EpsilonGreedy(n_actions=3, epsilon=1.0)
    selector.set_values([0.3, 0.8, 0.5])
    
    actions = [selector.choose() for _ in range(100)]
    unique_actions = set(actions)
    assert len(unique_actions) >= 2, "❌ With epsilon=1, should try different actions"
    print(f"✓ Tried actions: {unique_actions}\n")
    
    # Test 3: Balanced epsilon-greedy
    print("Test 3: Balanced epsilon-greedy (epsilon=0.2)")
    selector = EpsilonGreedy(n_actions=3, epsilon=0.2)
    selector.set_values([0.3, 0.8, 0.5])
    
    actions = [selector.choose() for _ in range(1000)]
    
    # Count action frequencies
    action_counts = {i: actions.count(i) for i in range(3)}
    
    # Action 1 should be most common (it's best)
    assert action_counts[1] > action_counts[0], "❌ Best action should be chosen most often"
    assert action_counts[1] > action_counts[2], "❌ Best action should be chosen most often"
    
    # But other actions should still appear (exploration)
    assert action_counts[0] > 0, "❌ Should occasionally explore action 0"
    assert action_counts[2] > 0, "❌ Should occasionally explore action 2"
    
    print(f"✓ Action frequencies: {action_counts}")
    print(f"  Action 1 chosen ~{action_counts[1]/10:.0f}% of time (expected ~87%)\n")
    
    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    test_epsilon_greedy()


# ===========================================================================
# HINTS
# ===========================================================================
"""
HINT 1 - Initialization:
    self.n_actions = n_actions
    self.epsilon = epsilon
    self.values = np.zeros(n_actions)

HINT 2 - Set values:
    self.values = np.array(values)

HINT 3 - Choose action:
    if np.random.random() < self.epsilon:
        # Explore: random action
        return np.random.randint(self.n_actions)
    else:
        # Exploit: best action
        return np.argmax(self.values)
"""
