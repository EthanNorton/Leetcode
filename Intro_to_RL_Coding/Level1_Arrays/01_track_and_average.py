"""
Problem: Track Rewards and Calculate Average

Difficulty: Easy
Category: Arrays & Basic Math

Description:
You're tracking rewards from a single action. Each time you take the action,
you get a reward. Your job is to keep track of:
1. How many times you've taken the action
2. The total reward received
3. The average reward

This is the CORE of RL value estimation!

Example:
    tracker = RewardTracker()
    tracker.add_reward(1.0)  # First reward
    tracker.add_reward(0.5)  # Second reward
    tracker.add_reward(0.8)  # Third reward
    
    print(tracker.get_count())    # Should print: 3
    print(tracker.get_total())    # Should print: 2.3
    print(tracker.get_average())  # Should print: 0.766...

Learning Objectives:
1. Track values with simple variables
2. Calculate running average
3. Handle edge cases (division by zero)
"""

class RewardTracker:
    def __init__(self):
        """
        Initialize the tracker
        
        TODO: Create variables to store:
        - count: how many rewards we've seen (starts at 0)
        - total: sum of all rewards (starts at 0.0)
        """
        # YOUR CODE HERE
        pass
        
    def add_reward(self, reward):
        """
        Add a new reward to our tracking
        
        Args:
            reward (float): The reward value to add
            
        TODO: Update count and total
        """
        # YOUR CODE HERE
        pass
        
    def get_count(self):
        """
        Return how many rewards we've tracked
        
        Returns:
            int: Number of rewards
            
        TODO: Return the count
        """
        # YOUR CODE HERE
        pass
        
    def get_total(self):
        """
        Return the total sum of all rewards
        
        Returns:
            float: Sum of all rewards
            
        TODO: Return the total
        """
        # YOUR CODE HERE
        pass
        
    def get_average(self):
        """
        Calculate and return the average reward
        
        Returns:
            float: Average reward (or 0 if no rewards yet)
            
        TODO: Calculate average = total / count
        HINT: What if count is 0? Return 0 to avoid division by zero
        """
        # YOUR CODE HERE
        pass


# ===========================================================================
# TESTS - Run these to check your code!
# ===========================================================================

def test_reward_tracker():
    print("Testing RewardTracker...\n")
    
    tracker = RewardTracker()
    
    # Test 1: Initial state
    print("Test 1: Initial state")
    assert tracker.get_count() == 0, "❌ Initial count should be 0"
    assert tracker.get_total() == 0.0, "❌ Initial total should be 0.0"
    assert tracker.get_average() == 0.0, "❌ Initial average should be 0.0"
    print("✓ Passed!\n")
    
    # Test 2: Adding one reward
    print("Test 2: Adding one reward")
    tracker.add_reward(1.0)
    assert tracker.get_count() == 1, "❌ Count should be 1 after adding one reward"
    assert tracker.get_total() == 1.0, "❌ Total should be 1.0"
    assert tracker.get_average() == 1.0, "❌ Average should be 1.0"
    print("✓ Passed!\n")
    
    # Test 3: Adding more rewards
    print("Test 3: Adding more rewards")
    tracker.add_reward(0.5)
    tracker.add_reward(0.9)
    assert tracker.get_count() == 3, "❌ Count should be 3"
    assert abs(tracker.get_total() - 2.4) < 0.01, "❌ Total should be 2.4"
    assert abs(tracker.get_average() - 0.8) < 0.01, "❌ Average should be 0.8"
    print("✓ Passed!\n")
    
    # Test 4: Many rewards
    print("Test 4: Adding many rewards")
    for _ in range(7):
        tracker.add_reward(1.0)
    assert tracker.get_count() == 10, "❌ Count should be 10"
    print("✓ Passed!\n")
    
    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print(f"\nFinal state:")
    print(f"  Count: {tracker.get_count()}")
    print(f"  Total: {tracker.get_total():.2f}")
    print(f"  Average: {tracker.get_average():.3f}")


if __name__ == "__main__":
    test_reward_tracker()


# ===========================================================================
# HINTS (Don't look until you've tried for 10 minutes!)
# ===========================================================================
"""
HINT 1 - Initialization:
In __init__, you need:
    self.count = 0
    self.total = 0.0

HINT 2 - Adding rewards:
In add_reward, you need:
    self.total += reward  # Add to total
    self.count += 1       # Increment count

HINT 3 - Getting values:
In get_count:
    return self.count
In get_total:
    return self.total

HINT 4 - Average calculation:
In get_average:
    if self.count == 0:
        return 0.0
    return self.total / self.count
"""
