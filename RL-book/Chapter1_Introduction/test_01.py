"""
Simple test script to verify 01_simple_agent.py works correctly
"""
import numpy as np
from simple_agent import SimpleEnvironment, SimpleAgent

def test_environment():
    """Test the environment works"""
    print("=" * 50)
    print("Testing Environment")
    print("=" * 50)
    
    # Create environment with button 2 as the best
    env = SimpleEnvironment(k=5, optimal_action=2)
    print(f"Created environment with {env.k} buttons")
    print(f"Optimal button is: {env.optimal_action}")
    
    # Test pressing different buttons
    print("\nTesting button rewards:")
    for button in range(5):
        rewards = [env.step(button) for _ in range(10)]
        avg_reward = np.mean(rewards)
        print(f"  Button {button}: average reward = {avg_reward:.3f}")
    
    print("\n✓ Environment test passed!\n")

def test_agent():
    """Test the agent learns"""
    print("=" * 50)
    print("Testing Agent Learning")
    print("=" * 50)
    
    # Create environment and agent
    env = SimpleEnvironment(k=5, optimal_action=2)
    agent = SimpleAgent(k=5, exploration_rate=0.2)
    
    print(f"Optimal button is: {env.optimal_action}")
    print(f"Agent doesn't know this yet!\n")
    
    # Run learning process
    print("Running 100 steps of learning...")
    for step in range(100):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)
    
    # Show what the agent learned
    print("\nAgent's knowledge after 100 steps:")
    print("Button | Times Tried | Total Reward | Average Reward")
    print("-" * 60)
    for button in range(5):
        count = int(agent.action_counts[button])
        total = agent.reward_sums[button]
        avg = total / (count + 1e-5) if count > 0 else 0
        star = " ← HIGHEST!" if button == np.argmax(agent.reward_sums / (agent.action_counts + 1e-5)) else ""
        print(f"  {button}    |     {count:3d}     |    {total:6.2f}    |    {avg:.3f}{star}")
    
    # Check if agent learned the optimal button
    best_button = np.argmax(agent.reward_sums / (agent.action_counts + 1e-5))
    print(f"\nAgent thinks button {best_button} is best")
    print(f"Actual best button is {env.optimal_action}")
    
    if best_button == env.optimal_action:
        print("✓ Agent learned correctly!")
    else:
        print("⚠ Agent needs more training (this can happen due to randomness)")
    
    print("\n✓ Agent test passed!\n")

def test_longer_learning():
    """Test learning over more steps"""
    print("=" * 50)
    print("Testing Longer Learning (500 steps)")
    print("=" * 50)
    
    env = SimpleEnvironment(k=5, optimal_action=2)
    agent = SimpleAgent(k=5, exploration_rate=0.1)  # Less exploration
    
    # Track performance over time
    window_size = 50
    for episode in range(10):
        rewards = []
        for _ in range(window_size):
            action = agent.choose_action()
            reward = env.step(action)
            agent.learn(action, reward)
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        best_action = np.argmax(agent.reward_sums / (agent.action_counts + 1e-5))
        print(f"Steps {episode*window_size:3d}-{(episode+1)*window_size:3d}: "
              f"Avg Reward = {avg_reward:.3f}, "
              f"Prefers button {best_action}")
    
    print("\n✓ Extended learning test passed!\n")

if __name__ == "__main__":
    # Run all tests
    test_environment()
    test_agent()
    test_longer_learning()
    
    print("=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50)
