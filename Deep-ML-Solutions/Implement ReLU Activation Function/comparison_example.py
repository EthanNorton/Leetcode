"""
Demonstration: Why Zeroing Negatives is Usually OK (But Can Be a Problem)
"""

def relu(x):
    """Standard ReLU: negatives become 0"""
    return max(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: negatives become small positive"""
    return max(alpha * x, x)

# Example: Neural network layer
print("=" * 70)
print("WHY ZEROING NEGATIVES IS USUALLY OK")
print("=" * 70)

print("\nExample 1: Image Classification - Cat vs Dog")
print("-" * 70)

# Simulated neuron outputs (before ReLU)
neurons = {
    "cat_detector": 8.5,      # Strong positive = "I see a cat!"
    "dog_detector": -3.2,    # Negative = "I don't see a dog"
    "car_detector": -1.5,    # Negative = "I don't see a car"
    "bird_detector": 0.3     # Weak positive = "Maybe a bird?"
}

print("Neuron outputs (before activation):")
for name, value in neurons.items():
    print(f"  {name:15s}: {value:6.2f}")

print("\nAfter Standard ReLU:")
for name, value in neurons.items():
    activated = relu(value)
    status = "ACTIVE" if activated > 0 else "INACTIVE"
    print(f"  {name:15s}: {value:6.2f} -> {activated:6.2f} ({status})")

print("\nInterpretation:")
print("  - cat_detector: ACTIVE (8.5) -> 'Yes, I see a cat!'")
print("  - dog_detector: INACTIVE (0) -> 'No dog here' (correct!)")
print("  - car_detector: INACTIVE (0) -> 'No car here' (correct!)")
print("  - bird_detector: ACTIVE (0.3) -> 'Maybe a bird?'")
print("\n  Result: Network correctly identifies it's a cat!")

print("\n" + "=" * 70)
print("WHEN ZEROING NEGATIVES CAN BE A PROBLEM")
print("=" * 70)

print("\nExample 2: The 'Dying ReLU' Problem")
print("-" * 70)

# Simulate a neuron that keeps getting negative outputs
print("Neuron learning over time:")
print("  Iteration 1: output = -2.0 -> ReLU -> 0.0 (gradient = 0, can't learn!)")
print("  Iteration 2: output = -1.8 -> ReLU -> 0.0 (still gradient = 0)")
print("  Iteration 3: output = -1.5 -> ReLU -> 0.0 (still stuck!)")
print("  ...")
print("  Iteration 100: output = -0.5 -> ReLU -> 0.0 (NEURON IS DEAD!)")
print("\n  Problem: Neuron can never recover because gradient is always 0!")

print("\nWith Leaky ReLU (alpha=0.01):")
print("  Iteration 1: output = -2.0 -> Leaky ReLU -> -0.02 (gradient = 0.01, can learn!)")
print("  Iteration 2: output = -1.8 -> Leaky ReLU -> -0.018 (still learning)")
print("  Iteration 3: output = -1.5 -> Leaky ReLU -> -0.015 (making progress)")
print("  ...")
print("  Iteration 100: output = 0.5 -> Leaky ReLU -> 0.5 (NEURON RECOVERED!)")
print("\n  Solution: Small gradient allows neuron to learn and recover!")

print("\n" + "=" * 70)
print("COMPARISON: Standard ReLU vs Leaky ReLU")
print("=" * 70)

test_values = [-5, -2, -1, 0, 1, 2, 5]

print("\nInput Value | Standard ReLU | Leaky ReLU (alpha=0.01)")
print("-" * 70)
for val in test_values:
    std_relu = relu(val)
    leaky = leaky_relu(val)
    print(f"    {val:6.1f}   |     {std_relu:6.2f}    |      {leaky:6.2f}")

print("\nKey Differences:")
print("  Standard ReLU: All negatives -> 0 (loses information, can cause 'dying ReLU')")
print("  Leaky ReLU: Negatives -> small values (preserves information, prevents 'dying ReLU')")

print("\n" + "=" * 70)
print("WHY ZEROING IS USUALLY OK")
print("=" * 70)

print("\n1. We WANT Some Neurons to Be Inactive")
print("   - Not every neuron should fire for every input")
print("   - Creates specialization (cat neuron fires for cats, not dogs)")
print("   - Sparse representation is efficient")

print("\n2. Network Learns to Produce Positive Values")
print("   - If a neuron is important, network learns weights to make it positive")
print("   - During training, important neurons become active")
print("   - Unimportant neurons stay inactive (which is fine!)")

print("\n3. Negatives Often Mean 'Not This Pattern'")
print("   - Positive = 'I see this pattern'")
print("   - Negative = 'I don't see this pattern'")
print("   - Zero = 'Definitely not' (clear, unambiguous signal)")

print("\n" + "=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print("Zeroing negatives is usually BENEFICIAL (creates sparsity, specialization)")
print("But it CAN cause problems (dying ReLU) in some cases")
print("Solution: Use Leaky ReLU when you need to preserve negative information")
print("=" * 70)

