"""
Problem: Simple 3-Button Agent (SIMPLIFIED VERSION)

Difficulty: Beginner

Description:
You have 3 buttons (numbered 0, 1, 2). One button gives the best rewards.
Your agent needs to figure out which button is best by trying them.

GOAL: Learn which button gives the highest average reward.

Learning Objectives:
1. Store rewards in arrays
2. Calculate averages
3. Choose between exploring (random) and exploiting (best known)
"""

import numpy as np

# ============================================================================
# PART 1: THE ENVIRONMENT (The "Game")
# ============================================================================

class SimpleEnvironment:
    """
    Think of this as a game with 3 buttons.
    One button is secretly the "best" button.
    """
    
    def __init__(self, optimal_action=None):
        """
        Set up the game
        
        Args:
            optimal_action: Which button is best (0, 1, or 2)
                          If None, we'll pick a random one
        """
        # We always have exactly 3 buttons
        self.k = 3
        
        # Pick which button is the "best" one
        if optimal_action is None:
            # Pick randomly: could be 0, 1, or 2
            self.optimal_action = np.random.randint(3)
        else:
            # Use the one you specified
            self.optimal_action = optimal_action
        
        print(f"🎮 Game created! One of the 3 buttons is best (it's a secret!)")
    
    def step(self, action):
        """
        Press a button and get a reward
        
        Args:
            action: Which button to press (0, 1, or 2)
        
        Returns:
            A number (the reward). Higher is better!
        """
        # Start with a base reward
        if action == self.optimal_action:
            # You pressed the best button! Big reward!
            reward = 1.0
        elif abs(action - self.optimal_action) == 1:
            # You pressed a button next to the best one. Medium reward.
            reward = 0.5
        else:
            # You pressed the worst button. Small reward.
            reward = 0.2
        
        # Add a tiny bit of randomness (so it's not too obvious)
        noise = np.random.normal(0, 0.1)  # Small random number
        
        return reward + noise


# ============================================================================
# PART 2: THE AGENT (The "Learner")
# ============================================================================

class SimpleAgent:
    """
    This is the agent that learns which button is best.
    It keeps track of rewards and makes decisions.
    """
    
    def __init__(self, exploration_rate=0.3):
        """
        Create an agent
        
        Args:
            exploration_rate: How often to try random buttons (0.3 = 30% of the time)
        """
        # We always work with 3 buttons
        self.k = 3
        
        # How often should we explore? (0.3 = 30% random, 70% best known)
        self.exploration_rate = exploration_rate
        
        # Create arrays to track information about each button
        # Index 0 = Button 0, Index 1 = Button 1, Index 2 = Button 2
        
        # Total rewards received from each button (starts at zero)
        self.reward_sums = np.zeros(3)  # [0, 0, 0]
        
        # How many times we've pressed each button (starts at zero)
        self.action_counts = np.zeros(3)  # [0, 0, 0]
        
        print(f"🤖 Agent created! Will explore {exploration_rate*100}% of the time")
    
    def choose_action(self):
        """
        Decide which button to press
        
        Returns:
            The button number to press (0, 1, or 2)
        """
        # Generate a random number between 0 and 1
        random_number = np.random.random()
        
        # Should we explore (try random) or exploit (try best known)?
        if random_number < self.exploration_rate:
            # EXPLORE: Try a random button
            action = np.random.randint(3)  # Picks 0, 1, or 2 randomly
            return action
        else:
            # EXPLOIT: Pick the button with the best average reward so far
            
            # Calculate average reward for each button
            # Average = Total Rewards / Number of Times Pressed
            # We add a tiny number (1e-5) to avoid dividing by zero at the start
            averages = self.reward_sums / (self.action_counts + 1e-5)
            
            # Find which button has the highest average
            # np.argmax returns the INDEX of the maximum value
            best_action = np.argmax(averages)
            
            return best_action
    
    def learn(self, action, reward):
        """
        Update our knowledge after pressing a button
        
        Args:
            action: Which button we pressed (0, 1, or 2)
            reward: What reward we got
        """
        # Add this reward to our running total for this button
        self.reward_sums[action] += reward
        
        # Increment the count for this button
        self.action_counts[action] += 1
    
    def get_averages(self):
        """
        Get the current average reward for each button
        
        Returns:
            Array of averages: [avg_button_0, avg_button_1, avg_button_2]
        """
        # Calculate and return averages
        return self.reward_sums / (self.action_counts + 1e-5)


# ============================================================================
# PART 3: TESTING AND RUNNING
# ============================================================================

def run_simple_demo():
    """
    Run a simple demo to see the agent learn
    """
    print("\n" + "="*70)
    print("SIMPLE 3-BUTTON LEARNING DEMO")
    print("="*70 + "\n")
    
    # Step 1: Create the environment (button 1 is best)
    env = SimpleEnvironment(optimal_action=1)
    print(f"✓ The best button is: {env.optimal_action} (secret!)\n")
    
    # Step 2: Create the agent
    agent = SimpleAgent(exploration_rate=0.3)
    print()
    
    # Step 3: Let the agent learn by trying buttons
    print("🎯 Starting learning process...\n")
    print("Button | Times Tried | Total Reward | Average")
    print("-" * 60)
    
    # Run 30 steps
    for step in range(30):
        # Agent picks a button
        action = agent.choose_action()
        
        # Press the button, get reward
        reward = env.step(action)
        
        # Agent learns from this
        agent.learn(action, reward)
        
        # Show progress every 10 steps
        if (step + 1) % 10 == 0:
            averages = agent.get_averages()
            for button in range(3):
                count = int(agent.action_counts[button])
                total = agent.reward_sums[button]
                avg = averages[button]
                star = " ← BEST SO FAR" if button == np.argmax(averages) else ""
                print(f"  {button}    |     {count:2d}      |    {total:6.2f}    | {avg:.3f}{star}")
            print("-" * 60)
    
    # Step 4: Show final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    averages = agent.get_averages()
    best_button = np.argmax(averages)
    
    print(f"\n🎯 Agent thinks button {best_button} is best")
    print(f"✓ Actual best button is {env.optimal_action}")
    
    if best_button == env.optimal_action:
        print("\n🎉 SUCCESS! The agent learned correctly!")
    else:
        print("\n⚠️  Agent needs more training (can happen with small sample)")
    
    print("\n" + "="*70)


def test_agent_environment():
    """
    Automated test function
    """
    print("Running automated tests...")
    
    # Test 1: Environment works
    env = SimpleEnvironment(optimal_action=1)
    reward = env.step(1)
    assert isinstance(reward, (int, float)), "Reward should be a number"
    print("✓ Environment test passed")
    
    # Test 2: Agent works
    agent = SimpleAgent()
    action = agent.choose_action()
    assert action in [0, 1, 2], "Action should be 0, 1, or 2"
    print("✓ Agent test passed")
    
    # Test 3: Learning works
    for _ in range(20):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)
    
    assert np.sum(agent.action_counts) == 20, "Should track all actions"
    print("✓ Learning test passed")
    
    print("\nAll tests passed! ✓")


# ============================================================================
# RUN THE CODE
# ============================================================================

if __name__ == "__main__":
    # Run the demo
    run_simple_demo()
    
    print("\n\n")
    
    # Run tests
    test_agent_environment()
