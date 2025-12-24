"""
ML Concepts Visualization Generator
===================================

Generate visual diagrams and plots to help understand ML concepts.
This creates visual aids for studying CMU MSML preparation.

Requirements: matplotlib, numpy
Install: pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "visualizations"
output_dir.mkdir(exist_ok=True)

def plot_relu_function():
    """Visualize ReLU activation function"""
    x = np.linspace(-5, 5, 100)
    y = np.maximum(0, x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='ReLU')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output f(x)', fontsize=12)
    plt.title('ReLU Activation Function: f(x) = max(0, x)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xlim(-5, 5)
    plt.ylim(-1, 5)
    
    # Add annotations
    plt.annotate('Negative inputs -> 0', xy=(-3, 0.2), fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    plt.annotate('Positive inputs pass through', xy=(2, 2.5), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'relu_function.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'relu_function.png'}")
    plt.close()

def plot_sigmoid_function():
    """Visualize Sigmoid activation function"""
    x = np.linspace(-5, 5, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r-', linewidth=2, label='Sigmoid')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='y=0.5')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output f(x)', fontsize=12)
    plt.title('Sigmoid Activation Function: f(x) = 1 / (1 + e^(-x))', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    
    # Add annotations
    plt.annotate('Output always 0-1\n(Probability range)', xy=(3, 0.9), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.annotate('S-shaped curve', xy=(0, 0.5), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sigmoid_function.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sigmoid_function.png'}")
    plt.close()

def plot_activation_comparison():
    """Compare ReLU, Sigmoid, and Softmax"""
    x = np.linspace(-5, 5, 100)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, relu, 'b-', linewidth=2, label='ReLU')
    plt.plot(x, sigmoid, 'r-', linewidth=2, label='Sigmoid')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output f(x)', fontsize=12)
    plt.title('Activation Functions Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xlim(-5, 5)
    plt.ylim(-1, 5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'activation_comparison.png'}")
    plt.close()

def plot_gradient_descent_convergence():
    """Visualize gradient descent convergence"""
    # Simulate gradient descent
    iterations = np.arange(0, 50)
    # Simulate loss decreasing (exponential decay)
    loss = 10 * np.exp(-iterations / 10) + 0.1
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss, 'b-o', linewidth=2, markersize=4, label='Loss')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (Error)', fontsize=12)
    plt.title('Gradient Descent: Loss Decreasing Over Iterations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Add annotations
    plt.annotate('Start: High error', xy=(0, 10), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    plt.annotate('Convergence: Low error', xy=(40, 0.2), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_descent_convergence.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gradient_descent_convergence.png'}")
    plt.close()

def plot_softmax_example():
    """Visualize Softmax converting scores to probabilities"""
    scores = np.array([2.0, 1.0, 0.1])
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)
    
    categories = ['Class 0', 'Class 1', 'Class 2']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scores
    ax1.bar(categories, scores, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Raw Scores', fontsize=12)
    ax1.set_title('Input: Raw Scores (Logits)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Probabilities
    ax2.bar(categories, probabilities, color='lightgreen', alpha=0.7)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Output: Probabilities (Sum to 1.0)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add probability labels
    for i, prob in enumerate(probabilities):
        ax2.text(i, prob + 0.02, f'{prob:.2%}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'softmax_example.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'softmax_example.png'}")
    plt.close()

def plot_feature_scaling_comparison():
    """Compare original vs scaled features"""
    # Original features (different scales)
    house_sizes = np.array([1000, 2000, 3000, 4000, 5000])
    bedrooms = np.array([1, 2, 3, 4, 5])
    
    # Scaled features
    sizes_scaled = (house_sizes - house_sizes.min()) / (house_sizes.max() - house_sizes.min())
    beds_scaled = (bedrooms - bedrooms.min()) / (bedrooms.max() - bedrooms.min())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original (different scales)
    x_pos = np.arange(len(house_sizes))
    width = 0.35
    ax1.bar(x_pos - width/2, house_sizes/1000, width, label='House Size (×1000 sqft)', alpha=0.7)
    ax1.bar(x_pos + width/2, bedrooms, width, label='Bedrooms', alpha=0.7)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Before Scaling: Different Scales!', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Sample {i+1}' for i in range(len(house_sizes))])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scaled (same scale)
    ax2.bar(x_pos - width/2, sizes_scaled, width, label='House Size (scaled)', alpha=0.7, color='skyblue')
    ax2.bar(x_pos + width/2, beds_scaled, width, label='Bedrooms (scaled)', alpha=0.7, color='lightgreen')
    ax2.set_ylabel('Scaled Value', fontsize=12)
    ax2.set_title('After Scaling: Same Scale!', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Sample {i+1}' for i in range(len(house_sizes))])
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_scaling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_scaling_comparison.png'}")
    plt.close()

def plot_matrix_multiplication_visual():
    """Visual diagram of matrix-vector multiplication"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Draw matrix A
    matrix_text = r'$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$'
    ax.text(0.2, 0.5, matrix_text, fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.2, 0.3, 'Matrix A\n(3×2)', fontsize=12, ha='center', va='top')
    
    # Draw × symbol
    ax.text(0.5, 0.5, '×', fontsize=30, ha='center', va='center')
    
    # Draw vector b
    vector_text = r'$\begin{bmatrix} 7 \\ 8 \end{bmatrix}$'
    ax.text(0.8, 0.5, vector_text, fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.8, 0.3, 'Vector b\n(2×1)', fontsize=12, ha='center', va='top')
    
    # Draw = symbol
    ax.text(0.5, 0.2, '=', fontsize=30, ha='center', va='center')
    
    # Draw result
    result_text = r'$\begin{bmatrix} 23 \\ 53 \\ 83 \end{bmatrix}$'
    ax.text(0.5, 0.05, result_text, fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.text(0.5, -0.15, 'Result\n(3×1)', fontsize=12, ha='center', va='top')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.8)
    plt.title('Matrix-Vector Multiplication Visual', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'matrix_multiplication_visual.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'matrix_multiplication_visual.png'}")
    plt.close()

def plot_reshape_visual():
    """Visual diagram of matrix reshaping"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Original
    ax1 = axes[0]
    ax1.axis('off')
    original = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    
    # Draw grid
    for i in range(2):
        for j in range(6):
            ax1.add_patch(plt.Rectangle((j, 1-i), 1, 1, fill=True, 
                                        facecolor='lightblue', edgecolor='black', linewidth=1))
            ax1.text(j+0.5, 1.5-i, str(original[i, j]), ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_title('Original: 2×6 Matrix', fontsize=14, fontweight='bold', pad=10)
    ax1.text(3, -0.3, 'Reading order: 1→2→3→4→5→6→7→8→9→10→11→12', 
             ha='center', fontsize=10, style='italic')
    
    # Reshaped
    ax2 = axes[1]
    ax2.axis('off')
    reshaped = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    
    # Draw grid
    for i in range(3):
        for j in range(4):
            ax2.add_patch(plt.Rectangle((j, 2-i), 1, 1, fill=True,
                                       facecolor='lightgreen', edgecolor='black', linewidth=1))
            ax2.text(j+0.5, 2.5-i, str(reshaped[i, j]), ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_title('Reshaped: 3×4 Matrix', fontsize=14, fontweight='bold', pad=10)
    ax2.text(2, -0.3, 'Same elements, different arrangement!', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reshape_visual.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'reshape_visual.png'}")
    plt.close()

def plot_one_hot_encoding_visual():
    """Visual diagram of one-hot encoding"""
    categories = ['red', 'blue', 'green', 'red', 'blue']
    encoded = [
        [1, 0, 0],  # red
        [0, 1, 0],  # blue
        [0, 0, 1],  # green
        [1, 0, 0],  # red
        [0, 1, 0]   # blue
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original categories
    ax1.barh(range(len(categories)), [1]*len(categories), color=['red', 'blue', 'green', 'red', 'blue'])
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels([f'Sample {i+1}: {cat}' for i, cat in enumerate(categories)])
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_title('Original: Categorical Data', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.2)
    
    # One-hot encoded
    colors_map = {'red': 'red', 'blue': 'blue', 'green': 'green'}
    x_pos = np.arange(len(categories))
    width = 0.25
    
    for i, (cat, encoding) in enumerate(zip(categories, encoded)):
        for j, (val, color) in enumerate(zip(encoding, ['red', 'blue', 'green'])):
            if val == 1:
                ax2.barh(i, 1, left=j, height=0.8, color=color, alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels([f'Sample {i+1}' for i in range(len(categories))])
    ax2.set_xlabel('One-Hot Encoded', fontsize=12)
    ax2.set_title('One-Hot Encoded: Binary Vectors', fontsize=13, fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Red', 'Blue', 'Green'])
    ax2.set_xlim(-0.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'one_hot_encoding_visual.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'one_hot_encoding_visual.png'}")
    plt.close()

def plot_gradient_descent_3d():
    """3D visualization of gradient descent on a loss surface"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a simple loss surface (bowl shape)
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple bowl-shaped loss function
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
    
    # Simulate gradient descent path
    path_x = [2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    path_y = [2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    path_z = [x**2 + y**2 for x, y in zip(path_x, path_y)]
    
    ax.plot(path_x, path_y, path_z, 'r-o', linewidth=2, markersize=6, label='Gradient Descent Path')
    ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], color='red', s=100, marker='*', label='Start')
    ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], color='green', s=100, marker='*', label='Minimum')
    
    ax.set_xlabel('Parameter θ₁', fontsize=11)
    ax.set_ylabel('Parameter θ₂', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title('Gradient Descent: Finding Minimum on Loss Surface', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_descent_3d.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gradient_descent_3d.png'}")
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations"""
    print("=" * 70)
    print("GENERATING ML CONCEPT VISUALIZATIONS")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerating visualizations...\n")
    
    try:
        plot_relu_function()
        plot_sigmoid_function()
        plot_activation_comparison()
        plot_gradient_descent_convergence()
        plot_softmax_example()
        plot_feature_scaling_comparison()
        plot_matrix_multiplication_visual()
        plot_reshape_visual()
        plot_one_hot_encoding_visual()
        plot_gradient_descent_3d()
        
        print("\n" + "=" * 70)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nCheck the '{output_dir}' folder for all images.")
        print("\nGenerated files:")
        for file in sorted(output_dir.glob('*.png')):
            print(f"  - {file.name}")
        
    except ImportError as e:
        print(f"\nError: Missing required library. Install with: pip install matplotlib numpy")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nError generating visualizations: {e}")

if __name__ == "__main__":
    generate_all_visualizations()

