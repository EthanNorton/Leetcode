"""
Interactive Guide to Understanding ReLU
Run this file to see ReLU in action with visual examples!
"""

def relu(z: float) -> float:
    """ReLU activation function: max(0, z)"""
    return max(0, z)


def demonstrate_relu():
    """Walk through ReLU with examples"""
    
    print("=" * 60)
    print("RELU ACTIVATION FUNCTION - INTERACTIVE GUIDE")
    print("=" * 60)
    print()
    
    print("📚 What is ReLU?")
    print("   ReLU stands for 'Rectified Linear Unit'")
    print("   It's a simple rule: if input is positive, keep it; if negative, make it 0")
    print()
    
    print("🔍 The Formula:")
    print("   relu(x) = max(0, x)")
    print()
    
    print("=" * 60)
    print("EXAMPLES:")
    print("=" * 60)
    print()
    
    # Example 1: Positive number
    test1 = 5
    result1 = relu(test1)
    print(f"Example 1: relu({test1})")
    print(f"  Input: {test1} (positive number)")
    print(f"  Process: max(0, {test1}) = {result1}")
    print(f"  Output: {result1} ✓ (positive values pass through)")
    print()
    
    # Example 2: Negative number
    test2 = -3
    result2 = relu(test2)
    print(f"Example 2: relu({test2})")
    print(f"  Input: {test2} (negative number)")
    print(f"  Process: max(0, {test2}) = {result2}")
    print(f"  Output: {result2} ✓ (negative values become 0)")
    print()
    
    # Example 3: Zero
    test3 = 0
    result3 = relu(test3)
    print(f"Example 3: relu({test3})")
    print(f"  Input: {test3} (zero)")
    print(f"  Process: max(0, {test3}) = {result3}")
    print(f"  Output: {result3} ✓ (zero stays zero)")
    print()
    
    # Example 4: Multiple values
    print("=" * 60)
    print("VISUAL REPRESENTATION:")
    print("=" * 60)
    print()
    print("Input values:  [-3, -2, -1, 0, 1, 2, 3]")
    inputs = [-3, -2, -1, 0, 1, 2, 3]
    outputs = [relu(x) for x in inputs]
    print(f"ReLU outputs: {outputs}")
    print()
    print("Notice how all negative values become 0!")
    print()
    
    # Why it's useful
    print("=" * 60)
    print("WHY IS RELU IMPORTANT?")
    print("=" * 60)
    print()
    print("1. 🚀 Fast: Just a simple comparison (max operation)")
    print("2. 🧠 Non-linear: Helps neural networks learn complex patterns")
    print("3. 📈 Solves vanishing gradient problem (unlike sigmoid)")
    print("4. 🎯 Creates sparsity: Many neurons output 0 (efficient!)")
    print()
    
    # Try it yourself
    print("=" * 60)
    print("TRY IT YOURSELF:")
    print("=" * 60)
    print()
    print("You can test any number:")
    print("  relu(10) =", relu(10))
    print("  relu(-5) =", relu(-5))
    print("  relu(0.5) =", relu(0.5))
    print("  relu(-0.1) =", relu(-0.1))
    print()
    
    print("=" * 60)
    print("🎉 You now understand ReLU!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_relu()

