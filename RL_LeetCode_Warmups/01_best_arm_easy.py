"""
LeetCode #1: Find Best Performing Action

Difficulty: Easy
Similar to: LeetCode 121 (Best Time to Buy Stock)

Problem:
Given an array of average rewards for different actions, return the action
with the highest reward. If there are ties, return the smallest action number.

This is what argmax() does in RL, but implement it yourself!

Example 1:
    Input: rewards = [0.5, 0.8, 0.3, 0.9, 0.7]
    Output: 3
    Explanation: Action 3 has reward 0.9, which is highest

Example 2:
    Input: rewards = [1.0, 1.0, 0.5]
    Output: 0
    Explanation: Actions 0 and 1 tie at 1.0, return smaller index

Constraints:
- 1 <= len(rewards) <= 1000
- 0.0 <= rewards[i] <= 10.0
"""

def find_best_action(rewards):
    """
    Find action with highest reward
    
    Args:
        rewards (list): Reward values for each action
        
    Returns:
        int: Index of best action
        
    TODO: Implement this function
    HINT: Track the best value and its index as you iterate
    """
    # YOUR CODE HERE
    pass


# Test cases
def test():
    # Test 1
    assert find_best_action([0.5, 0.8, 0.3, 0.9, 0.7]) == 3
    print("✓ Test 1 passed")
    
    # Test 2
    assert find_best_action([1.0, 1.0, 0.5]) == 0
    print("✓ Test 2 passed")
    
    # Test 3
    assert find_best_action([0.1]) == 0
    print("✓ Test 3 passed")
    
    # Test 4
    assert find_best_action([5.0, 2.0, 8.0, 8.0, 3.0]) == 2
    print("✓ Test 4 passed")
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test()


# Solution approach (don't peek!)
"""
SOLUTION:
def find_best_action(rewards):
    best_action = 0
    best_reward = rewards[0]
    
    for i in range(1, len(rewards)):
        if rewards[i] > best_reward:
            best_reward = rewards[i]
            best_action = i
    
    return best_action

Time: O(n)
Space: O(1)
"""
