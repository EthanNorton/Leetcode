"""
Problem: Track Multiple Actions with Arrays

Difficulty: Easy
Category: Numpy Arrays

Description:
Now you have 3 different actions (like 3 different buttons).
You need to track rewards for EACH action separately using numpy arrays.

This is exactly what Q-learning and other RL algorithms do!

Example:
    tracker = MultiActionTracker(n_actions=3)
    tracker.add_reward(action=0, reward=1.0)
    tracker.add_reward(action=1, reward=0.5)
    tracker.add_reward(action=0, reward=0.8)
    
    print(tracker.get_average(action=0))  # Average for action 0
    print(tracker.get_best_action())      # Which action is best?

Learning Objectives:
1. Use numpy arrays to store values
2. Index arrays with action numbers
3. Find maximum values in arrays
"""

import numpy as np

class MultiActionTracker:
    def __init__(self, n_actions=3):
        """
        Initialize tracker for multiple actions
        
        Args:
            n_actions (int): Number of different actions to track
            
        TODO: Create numpy arrays to store:
        - counts: how many times each action was taken
        - totals: sum of rewards for each action
        
        HINT: Use np.zeros(n_actions) to create array of zeros
        """
        self.n_actions = n_actions
        # YOUR CODE HERE
        pass
        
    def add_reward(self, action, reward):
        """
        Add a reward for a specific action
        
        Args:
            action (int): Which action (0 to n_actions-1)
            reward (float): The reward received
            
        TODO: Update the counts and totals arrays for this action
        HINT: Use array indexing like self.totals[action] += reward
        """
        # YOUR CODE HERE
        pass
        
    def get_average(self, action):
        """
        Get average reward for a specific action
        
        Args:
            action (int): Which action to get average for
            
        Returns:
            float: Average reward for this action
            
        TODO: Calculate total / count for this action
        HINT: Check if count is 0 to avoid division by zero
        """
        # YOUR CODE HERE
        pass
        
    def get_all_averages(self):
        """
        Get averages for ALL actions at once
        
        Returns:
            array: Average rewards for each action
            
        TODO: Calculate averages for all actions
        HINT: You can divide arrays: self.totals / (self.counts + 1e-5)
        The 1e-5 prevents division by zero
        """
        # YOUR CODE HERE
        pass
        
    def get_best_action(self):
        """
        Find which action has the highest average reward
        
        Returns:
            int: The action with highest average
            
        TODO: Find the action with maximum average
        HINT: Use np.argmax() on the averages array
        """
        # YOUR CODE HERE
        pass


# ===========================================================================
# TESTS
# ===========================================================================

def test_multi_action_tracker():
    print("Testing MultiActionTracker...\n")
    
    tracker = MultiActionTracker(n_actions=3)
    
    # Test 1: Initialization
    print("Test 1: Initialization")
    assert hasattr(tracker, 'counts'), "❌ Need to create 'counts' array"
    assert hasattr(tracker, 'totals'), "❌ Need to create 'totals' array"
    assert len(tracker.counts) == 3, "❌ counts should have length 3"
    assert len(tracker.totals) == 3, "❌ totals should have length 3"
    print("✓ Arrays created correctly\n")
    
    # Test 2: Adding rewards
    print("Test 2: Adding rewards")
    tracker.add_reward(0, 1.0)
    tracker.add_reward(1, 0.5)
    tracker.add_reward(0, 0.8)
    print("✓ Rewards added successfully\n")
    
    # Test 3: Get average for specific action
    print("Test 3: Get average for action 0")
    avg0 = tracker.get_average(0)
    expected = (1.0 + 0.8) / 2  # Two rewards for action 0
    assert abs(avg0 - expected) < 0.01, f"❌ Expected {expected}, got {avg0}"
    print(f"✓ Average for action 0: {avg0:.3f}\n")
    
    # Test 4: Get all averages
    print("Test 4: Get all averages")
    averages = tracker.get_all_averages()
    assert len(averages) == 3, "❌ Should return array of length 3"
    print(f"✓ All averages: {averages}\n")
    
    # Test 5: Find best action
    print("Test 5: Find best action")
    tracker.add_reward(2, 2.0)  # Make action 2 clearly best
    tracker.add_reward(2, 1.8)
    best = tracker.get_best_action()
    assert best == 2, f"❌ Action 2 should be best, got {best}"
    print(f"✓ Best action: {best}\n")
    
    # Show final state
    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nFinal state:")
    averages = tracker.get_all_averages()
    for action in range(3):
        count = int(tracker.counts[action])
        avg = averages[action]
        best_marker = " ★ BEST" if action == tracker.get_best_action() else ""
        print(f"  Action {action}: {count} tries, avg reward = {avg:.3f}{best_marker}")


if __name__ == "__main__":
    test_multi_action_tracker()


# ===========================================================================
# HINTS
# ===========================================================================
"""
HINT 1 - Initialization:
    self.counts = np.zeros(n_actions)
    self.totals = np.zeros(n_actions)

HINT 2 - Adding rewards:
    self.totals[action] += reward
    self.counts[action] += 1

HINT 3 - Get average for one action:
    if self.counts[action] == 0:
        return 0.0
    return self.totals[action] / self.counts[action]

HINT 4 - Get all averages:
    return self.totals / (self.counts + 1e-5)

HINT 5 - Get best action:
    averages = self.get_all_averages()
    return np.argmax(averages)
"""
