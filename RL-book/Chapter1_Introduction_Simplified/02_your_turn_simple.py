"""
YOUR TURN: Build Your Own Simple Agent

Difficulty: Beginner (With Guidance)

Description:
Now it's your turn! Fill in the missing parts (marked with TODO).
The structure is provided, you just need to add the logic.

TASK: Complete the missing parts to make a working agent-environment system.
"""

import numpy as np

class ThreeButtonEnvironment:
    """
    A game with 3 buttons. You need to find which one is best!
    """
    
    def __init__(self):
        self.k = 3
        # TODO: Pick a random optimal action (0, 1, or 2)
        # HINT: Use np.random.randint(3)
        self.optimal_action = None  # Replace None with your code
        
    def step(self, action):
        """Press a button and get a reward"""
        # TODO: Give reward based on which button was pressed
        # HINT: Best button gets 1.0, next best gets 0.5, worst gets 0.2
        
        if action == self.optimal_action:
            reward = None  # Replace None - what reward for best button?
        else:
            reward = None  # Replace None - what reward for other buttons?
        
        # Add small random noise
        noise = np.random.normal(0, 0.1)
        return reward + noise


class ThreeButtonAgent:
    """
    An agent that learns which button is best
    """
    
    def __init__(self):
        self.k = 3
        self.exploration_rate = 0.3
        
        # TODO: Create arrays to track rewards and counts
        # HINT: Use np.zeros(3) to create array [0, 0, 0]
        self.reward_sums = None  # Replace None
        self.action_counts = None  # Replace None
        
    def choose_action(self):
        """Pick which button to press"""
        
        # Generate random number
        random_val = np.random.random()
        
        if random_val < self.exploration_rate:
            # TODO: Return a random action (0, 1, or 2)
            # HINT: Use np.random.randint(3)
            return None  # Replace None
        else:
            # TODO: Calculate averages and return best action
            # HINT: averages = self.reward_sums / (self.action_counts + 1e-5)
            # HINT: Then use np.argmax(averages)
            averages = None  # Calculate averages here
            return None  # Return best action here
    
    def learn(self, action, reward):
        """Learn from the reward we got"""
        
        # TODO: Update reward_sums for this action
        # HINT: Add reward to current sum: self.reward_sums[action] += reward
        pass  # Replace pass with your code
        
        # TODO: Update action_counts for this action
        # HINT: Add 1 to current count: self.action_counts[action] += 1
        pass  # Replace pass with your code


# ============================================================================
# TESTING YOUR CODE
# ============================================================================

def test_your_implementation():
    """
    Test if your implementation works
    """
    print("Testing your implementation...\n")
    
    # Create environment and agent
    env = ThreeButtonEnvironment()
    agent = ThreeButtonAgent()
    
    # Check if optimal action is set
    assert env.optimal_action is not None, "❌ Set optimal_action in ThreeButtonEnvironment.__init__"
    assert env.optimal_action in [0, 1, 2], "❌ optimal_action should be 0, 1, or 2"
    print("✓ Environment initialized correctly")
    
    # Check if arrays are initialized
    assert agent.reward_sums is not None, "❌ Initialize reward_sums array"
    assert agent.action_counts is not None, "❌ Initialize action_counts array"
    assert len(agent.reward_sums) == 3, "❌ reward_sums should have length 3"
    assert len(agent.action_counts) == 3, "❌ action_counts should have length 3"
    print("✓ Agent arrays initialized correctly")
    
    # Test choosing action
    action = agent.choose_action()
    assert action is not None, "❌ choose_action should return a value"
    assert action in [0, 1, 2], "❌ Action should be 0, 1, or 2"
    print("✓ Agent can choose actions")
    
    # Test learning
    initial_sum = agent.reward_sums[action].copy()
    initial_count = agent.action_counts[action].copy()
    
    reward = env.step(action)
    agent.learn(action, reward)
    
    assert agent.reward_sums[action] != initial_sum, "❌ reward_sums should update in learn()"
    assert agent.action_counts[action] != initial_count, "❌ action_counts should update in learn()"
    print("✓ Agent learning works")
    
    # Run full learning loop
    print("\n" + "="*60)
    print("Running learning loop...")
    print("="*60)
    
    for step in range(30):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)
    
    # Check results
    averages = agent.reward_sums / (agent.action_counts + 1e-5)
    best_button = np.argmax(averages)
    
    print(f"\nResults after 30 steps:")
    print(f"Button | Tries | Average Reward")
    print("-" * 40)
    for i in range(3):
        count = int(agent.action_counts[i])
        avg = averages[i]
        star = " ★" if i == best_button else ""
        print(f"  {i}    |  {count:2d}   | {avg:.3f}{star}")
    
    print(f"\n🎯 Agent's best button: {best_button}")
    print(f"✓ Actual best button: {env.optimal_action}")
    
    if best_button == env.optimal_action:
        print("\n🎉 SUCCESS! Your implementation works!")
    else:
        print("\n⚠️  Agent needs more steps (or got unlucky with randomness)")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    test_your_implementation()
