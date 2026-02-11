"""
Lecture 1 Challenge: Fibonacci Comparison
Dynamic Programming Learning Path - MIT & CS50 Lectures

This module implements and compares different approaches to computing Fibonacci numbers:
1. Naive Recursive (O(2^n) time)
2. Memoized Recursive (O(n) time, O(n) space)
3. Tabulated Iterative (O(n) time, O(n) space)
4. Space-Optimized (O(n) time, O(1) space)
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import sys
from functools import lru_cache

class FibonacciCalculator:
    """
    A comprehensive Fibonacci calculator with multiple implementation approaches.
    """
    
    def __init__(self):
        """Initialize the calculator with memoization storage."""
        self.memo: Dict[int, int] = {}
        self.call_count: int = 0
        self.performance_data: Dict[str, List[Tuple[int, float]]] = {
            'naive': [],
            'memoized': [],
            'tabulated': [],
            'space_optimized': []
        }
    
    def reset_call_count(self):
        """Reset the recursive call counter."""
        self.call_count = 0
    
    def naive_fibonacci(self, n: int) -> int:
        """
        Basic recursive implementation - O(2^n) time complexity.
        
        Args:
            n: The nth Fibonacci number to compute
            
        Returns:
            The nth Fibonacci number
        """
        self.call_count += 1
        
        # Base cases
        if n <= 1:
            return n
        
        # Recursive case
        return self.naive_fibonacci(n - 1) + self.naive_fibonacci(n - 2)
    
    def memoized_fibonacci(self, n: int) -> int:
        """
        Top-down DP with memoization - O(n) time, O(n) space.
        
        Args:
            n: The nth Fibonacci number to compute
            
        Returns:
            The nth Fibonacci number
        """
        self.call_count += 1
        
        # Base cases
        if n <= 1:
            return n
        
        # Check if already computed
        if n in self.memo:
            return self.memo[n]
        
        # Compute and store result
        result = self.memoized_fibonacci(n - 1) + self.memoized_fibonacci(n - 2)
        self.memo[n] = result
        return result
    
    def tabulated_fibonacci(self, n: int) -> int:
        """
        Bottom-up DP with tabulation - O(n) time, O(n) space.
        
        Args:
            n: The nth Fibonacci number to compute
            
        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n
        
        # Create table to store results
        dp = [0] * (n + 1)
        dp[1] = 1
        
        # Fill table bottom-up
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def space_optimized_fibonacci(self, n: int) -> int:
        """
        DP with O(1) space complexity - O(n) time, O(1) space.
        
        Args:
            n: The nth Fibonacci number to compute
            
        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n
        
        # Only store the last two values
        prev2, prev1 = 0, 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def matrix_fibonacci(self, n: int) -> int:
        """
        Matrix exponentiation approach - O(log n) time complexity.
        
        Args:
            n: The nth Fibonacci number to compute
            
        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n
        
        def matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            """Multiply two 2x2 matrices."""
            return [
                [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
            ]
        
        def matrix_power(matrix: List[List[int]], power: int) -> List[List[int]]:
            """Raise matrix to the given power using binary exponentiation."""
            if power == 1:
                return matrix
            
            if power % 2 == 0:
                half_power = matrix_power(matrix, power // 2)
                return matrix_multiply(half_power, half_power)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, power - 1))
        
        # Fibonacci matrix: [[1, 1], [1, 0]]
        fib_matrix = [[1, 1], [1, 0]]
        result_matrix = matrix_power(fib_matrix, n)
        
        return result_matrix[0][1]
    
    def compare_performance(self, n_values: List[int]) -> Dict[str, Dict[int, float]]:
        """
        Compare all methods for different values of n.
        
        Args:
            n_values: List of n values to test
            
        Returns:
            Dictionary with timing results for each method
        """
        results = {
            'naive': {},
            'memoized': {},
            'tabulated': {},
            'space_optimized': {},
            'matrix': {}
        }
        
        for n in n_values:
            print(f"Testing n = {n}")
            
            # Test naive (only for small values)
            if n <= 30:
                self.reset_call_count()
                start_time = time.time()
                result = self.naive_fibonacci(n)
                naive_time = time.time() - start_time
                results['naive'][n] = naive_time
                print(f"  Naive: {result} in {naive_time:.6f}s ({self.call_count} calls)")
            
            # Test memoized
            self.memo.clear()
            self.reset_call_count()
            start_time = time.time()
            result = self.memoized_fibonacci(n)
            memoized_time = time.time() - start_time
            results['memoized'][n] = memoized_time
            print(f"  Memoized: {result} in {memoized_time:.6f}s ({self.call_count} calls)")
            
            # Test tabulated
            start_time = time.time()
            result = self.tabulated_fibonacci(n)
            tabulated_time = time.time() - start_time
            results['tabulated'][n] = tabulated_time
            print(f"  Tabulated: {result} in {tabulated_time:.6f}s")
            
            # Test space-optimized
            start_time = time.time()
            result = self.space_optimized_fibonacci(n)
            space_opt_time = time.time() - start_time
            results['space_optimized'][n] = space_opt_time
            print(f"  Space-optimized: {result} in {space_opt_time:.6f}s")
            
            # Test matrix (for larger values)
            if n <= 1000:
                start_time = time.time()
                result = self.matrix_fibonacci(n)
                matrix_time = time.time() - start_time
                results['matrix'][n] = matrix_time
                print(f"  Matrix: {result} in {matrix_time:.6f}s")
            
            print()
        
        return results
    
    def visualize_performance(self, max_n: int = 30):
        """
        Create performance comparison charts.
        
        Args:
            max_n: Maximum value of n to test
        """
        # Test values (smaller range for naive method)
        n_values = list(range(5, max_n + 1, 5))
        
        # Get performance data
        results = self.compare_performance(n_values)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Time complexity comparison
        methods = ['memoized', 'tabulated', 'space_optimized', 'matrix']
        colors = ['blue', 'green', 'red', 'purple']
        
        for method, color in zip(methods, colors):
            if method in results and results[method]:
                n_vals = list(results[method].keys())
                times = list(results[method].values())
                ax1.plot(n_vals, times, 'o-', color=color, label=method, linewidth=2)
        
        ax1.set_xlabel('n')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Time Complexity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Call count comparison (naive vs memoized)
        call_counts_naive = []
        call_counts_memoized = []
        n_vals_small = list(range(5, 26, 5))
        
        for n in n_vals_small:
            # Naive call count
            self.reset_call_count()
            self.naive_fibonacci(n)
            call_counts_naive.append(self.call_count)
            
            # Memoized call count
            self.memo.clear()
            self.reset_call_count()
            self.memoized_fibonacci(n)
            call_counts_memoized.append(self.call_count)
        
        ax2.plot(n_vals_small, call_counts_naive, 'ro-', label='Naive', linewidth=2)
        ax2.plot(n_vals_small, call_counts_memoized, 'bo-', label='Memoized', linewidth=2)
        ax2.set_xlabel('n')
        ax2.set_ylabel('Number of Recursive Calls')
        ax2.set_title('Recursive Call Count Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Space complexity visualization
        space_complexity = {
            'naive': [n for n in n_vals_small],  # O(n) due to recursion stack
            'memoized': [n for n in n_values],   # O(n) for memo table
            'tabulated': [n for n in n_values],  # O(n) for DP table
            'space_optimized': [1 for _ in n_values],  # O(1)
            'matrix': [1 for _ in n_values]      # O(1)
        }
        
        for method, color in zip(['memoized', 'tabulated', 'space_optimized', 'matrix'], colors):
            if method == 'memoized' or method == 'tabulated':
                ax3.plot(n_values, space_complexity[method], 'o-', color=color, label=method, linewidth=2)
            else:
                ax3.plot(n_values, space_complexity[method], 'o-', color=color, label=method, linewidth=2)
        
        ax3.set_xlabel('n')
        ax3.set_ylabel('Space Complexity (units)')
        ax3.set_title('Space Complexity Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scalability test
        large_n_values = list(range(100, 1001, 100))
        matrix_times = []
        
        for n in large_n_values:
            start_time = time.time()
            self.matrix_fibonacci(n)
            matrix_times.append(time.time() - start_time)
        
        ax4.plot(large_n_values, matrix_times, 'go-', linewidth=2)
        ax4.set_xlabel('n')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Matrix Exponentiation Scalability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_complexity(self):
        """
        Analyze and print complexity analysis for all methods.
        """
        print("Complexity Analysis")
        print("=" * 50)
        print("Method              | Time    | Space   | Best Use Case")
        print("-" * 50)
        print("Naive Recursive     | O(2^n)  | O(n)    | Educational only")
        print("Memoized Recursive  | O(n)    | O(n)    | Top-down problems")
        print("Tabulated Iterative | O(n)    | O(n)    | Bottom-up problems")
        print("Space Optimized     | O(n)    | O(1)    | Memory-constrained")
        print("Matrix Exponentiation| O(log n)| O(1)    | Very large n")
        print()
        
        print("Key Insights:")
        print("- Memoization eliminates redundant calculations")
        print("- Tabulation avoids recursion overhead")
        print("- Space optimization trades readability for memory")
        print("- Matrix method is optimal for very large n")


def main():
    """
    Main function to demonstrate the FibonacciCalculator.
    """
    print("Fibonacci Calculator - Lecture 1 Challenge")
    print("=" * 50)
    
    # Initialize calculator
    calc = FibonacciCalculator()
    
    # Test basic functionality
    print("Basic Functionality Test")
    print("-" * 30)
    test_values = [0, 1, 5, 10, 20]
    
    for n in test_values:
        naive = calc.naive_fibonacci(n) if n <= 20 else "N/A (too slow)"
        memoized = calc.memoized_fibonacci(n)
        tabulated = calc.tabulated_fibonacci(n)
        space_opt = calc.space_optimized_fibonacci(n)
        matrix = calc.matrix_fibonacci(n)
        
        print(f"F({n:2d}) = {memoized:8d} | All methods agree: {memoized == tabulated == space_opt == matrix}")
    
    print("\n" + "=" * 50)
    
    # Performance comparison
    print("Performance Comparison")
    print("-" * 30)
    calc.compare_performance([10, 20, 30, 40, 50])
    
    # Complexity analysis
    calc.analyze_complexity()
    
    # Visualization
    print("Generating performance visualizations...")
    calc.visualize_performance(max_n=30)
    
    print("\nChallenge completed! 🎉")
    print("You've successfully implemented and compared multiple Fibonacci approaches!")


if __name__ == "__main__":
    main()
