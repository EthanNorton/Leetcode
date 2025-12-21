"""
Demonstration: Why Sparsity Matters
"""

import numpy as np

def relu(x):
    """Standard ReLU"""
    return max(0, x)

def apply_relu_layer(layer_output):
    """Apply ReLU to a layer"""
    return [relu(x) for x in layer_output]

# Example 1: Computational Efficiency
print("=" * 70)
print("EXAMPLE 1: COMPUTATIONAL EFFICIENCY")
print("=" * 70)

print("\nDense Network (all neurons somewhat active):")
dense_output = [0.3, 0.7, 0.5, 0.2, 0.9, 0.4, 0.6, 0.1]
print(f"  Output: {dense_output}")
print(f"  Active neurons: {len([x for x in dense_output if x > 0])}/8 (100%)")
print(f"  Computations needed: 8 (all neurons)")

print("\nSparse Network (few neurons active):")
sparse_output = [0, 0, 8.5, 0, 0, 0, 0.3, 0]
print(f"  Output: {sparse_output}")
print(f"  Active neurons: {len([x for x in sparse_output if x > 0])}/8 (25%)")
print(f"  Computations needed: 2 (only non-zero neurons)")
print(f"  Speed improvement: {8/2}x faster!")

# Example 2: How ReLU Creates Sparsity
print("\n" + "=" * 70)
print("EXAMPLE 2: HOW ReLU CREATES SPARSITY")
print("=" * 70)

print("\nLayer output BEFORE ReLU:")
before_relu = [2.5, -1.3, 0.8, -0.5, 5.2, -2.1, 0.1, -0.3]
print(f"  {before_relu}")
print(f"  Active (positive): {len([x for x in before_relu if x > 0])}/8")
print(f"  Inactive (negative): {len([x for x in before_relu if x < 0])}/8")

print("\nLayer output AFTER ReLU:")
after_relu = apply_relu_layer(before_relu)
print(f"  {after_relu}")
print(f"  Active (non-zero): {len([x for x in after_relu if x > 0])}/8")
print(f"  Inactive (zero): {len([x for x in after_relu if x == 0])}/8")
sparsity_percent = (len([x for x in after_relu if x == 0]) / len(after_relu)) * 100
print(f"  Sparsity: {sparsity_percent:.1f}% (zeros)")

# Example 3: Specialization
print("\n" + "=" * 70)
print("EXAMPLE 3: SPECIALIZATION")
print("=" * 70)

print("\nImage: Cat")
print("Dense network (confused):")
dense_cat = [0.3, 0.7, 0.5, 0.2, 0.9, 0.4, 0.6, 0.1]
print(f"  All neurons somewhat active: {dense_cat}")
print("  Problem: Can't tell what's important!")

print("\nSparse network (specialized):")
sparse_cat = [0, 0, 8.5, 0, 0, 0, 0.3, 0]
print(f"  Only 2 neurons active: {sparse_cat}")
print("  Neuron 2 (index 2): Detects 'cat ears' (8.5)")
print("  Neuron 6 (index 6): Detects 'whiskers' (0.3)")
print("  Others: Inactive (not relevant for cats)")
print("  Benefit: Clear specialization!")

print("\nImage: Dog")
sparse_dog = [0, 5.2, 0, 0, 0, 0.8, 0, 0]
print("Sparse network (different specialization):")
print(f"  Only 2 neurons active: {sparse_dog}")
print("  Neuron 1 (index 1): Detects 'dog ears' (5.2)")
print("  Neuron 5 (index 5): Detects 'dog snout' (0.8)")
print("  Different neurons than cat -> Clear distinction!")

# Example 4: Memory Efficiency
print("\n" + "=" * 70)
print("EXAMPLE 4: MEMORY EFFICIENCY")
print("=" * 70)

print("\nDense representation (1000 neurons, all active):")
print("  Need to store: 1000 float values")
print("  Memory: 1000 × 4 bytes = 4,000 bytes (4 KB)")

print("\nSparse representation (1000 neurons, only 100 active):")
print("  Can store: Only 100 non-zero values + 100 indices")
print("  Memory: ~800 bytes (5x less!)")
print("  Benefit: Can fit larger models in memory")

# Example 5: Real-world Impact
print("\n" + "=" * 70)
print("EXAMPLE 5: REAL-WORLD IMPACT")
print("=" * 70)

print("\nMobile Phone AI:")
print("  Without sparsity: Too slow, drains battery")
print("  With sparsity: Fast, efficient, practical!")

print("\nLarge Language Models (GPT):")
print("  Without sparsity: Billions of parameters = impractical")
print("  With sparsity: Can deploy efficiently")

print("\nComputer Vision:")
print("  Without sparsity: Can't do real-time processing")
print("  With sparsity: Real-time image recognition possible")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: WHY SPARSITY MATTERS")
print("=" * 70)
print("\n1. SPEED: Skip zero computations -> 10x faster")
print("2. MEMORY: Store only non-zeros -> 5x less space")
print("3. SPECIALIZATION: Each neuron has a clear job")
print("4. GENERALIZATION: Simpler models generalize better")
print("5. INTERPRETABILITY: Can see what network learned")
print("\nSparsity makes neural networks practical and deployable!")
print("=" * 70)

