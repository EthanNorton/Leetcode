"""
Simple test script - Just copy and paste this into your Python console or run it!
"""
import numpy as np
import sys
sys.path.append('.')

# Import from the file you created
exec(open('01_simple_agent.py').read())

def quick_test():
    """Quick test to see if everything works"""
    print("=" * 60)
    print("QUICK TEST - Simple Agent Environment")
    print("=" * 60)
    
    # Create environment with button 2 as the best
    env = SimpleEnvironment(k=5, optimal_action=2)
    print(f"\n✓ Created environment with {env.k} buttons")
    print(f"✓ Optimal button is: {env.optimal_action}\n")
    
    # Test pressing different buttons
    print("Testing button rewards (10 presses each):")
    print("-" * 60)
    for button in range(5):
        rewards = [env.step(button) for _ in range(10)]
        avg_reward = np.mean(rewards)
        status = "BEST! ★" if button == env.optimal_action else ""
        print(f"  Button {button}: average reward = {avg_reward:.3f}  {status}")
    
    print("\n" + "=" * 60)
    print("AGENT LEARNING TEST")
    print("=" * 60)
    
    # Create a new agent
    agent = SimpleAgent(k=5, exploration_rate=0.2)
    print(f"\n✓ Created agent (20% exploration rate)")
    print(f"✓ Agent will learn which button is best!\n")
    
    # Run learning for 200 steps
    print("Learning for 200 steps...")
    for step in range(200):
        action = agent.choose_action()
        reward = env.step(action)
        agent.learn(action, reward)
        
        # Print progress every 40 steps
        if (step + 1) % 40 == 0:
            best = np.argmax(agent.reward_sums / (agent.action_counts + 1e-5))
            print(f"  After {step+1:3d} steps: Agent prefers button {best}")
    
    # Show final results
    print("\n" + "-" * 60)
    print("FINAL RESULTS")
    print("-" * 60)
    print("\nWhat the agent learned:")
    print(f"{'Button':<10} {'Times Tried':<15} {'Avg Reward':<15} {'Status'}")
    print("-" * 60)
    
    best_button = np.argmax(agent.reward_sums / (agent.action_counts + 1e-5))
    
    for button in range(5):
        count = int(agent.action_counts[button])
        avg = (agent.reward_sums[button] / (count + 1e-5)) if count > 0 else 0
        
        status = ""
        if button == best_button:
            status += "FAVORITE"
        if button == env.optimal_action:
            status += " (OPTIMAL)"
        
        print(f"{button:<10} {count:<15} {avg:<15.3f} {status}")
    
    print("\n" + "=" * 60)
    if best_button == env.optimal_action:
        print("SUCCESS! Agent learned the optimal button! 🎉")
    else:
        print(f"Agent prefers button {best_button}, but optimal is {env.optimal_action}")
        print("(This can happen due to randomness - try running again!)")
    print("=" * 60)

if __name__ == "__main__":
    quick_test()
