"""
Lecture 1 Challenge: Convex Function Checker
Convex Optimization Learning Path - Stephen Boyd Lectures

This module implements a comprehensive convex function checker that can:
1. Test if a given function is convex using multiple criteria
2. Visualize convex vs non-convex functions
3. Handle both analytical and numerical approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import derivative
import sympy as sp
from typing import Callable, Tuple, List
import warnings

class ConvexFunctionChecker:
    """
    A comprehensive convex function checker with multiple testing methods.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the convex function checker.
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        
    def is_convex_analytical(self, func: Callable, domain: Tuple[float, float], 
                           num_points: int = 100) -> bool:
        """
        Check convexity using analytical methods (second-order condition).
        
        Args:
            func: Function to test (should be differentiable)
            domain: Tuple of (min, max) values for the domain
            num_points: Number of points to sample for testing
            
        Returns:
            bool: True if function appears convex, False otherwise
        """
        try:
            # Sample points in the domain
            x_values = np.linspace(domain[0], domain[1], num_points)
            
            # Check second derivative (if it exists)
            for x in x_values:
                try:
                    second_deriv = derivative(lambda t: derivative(func, t, dx=1e-6), 
                                            x, dx=1e-6)
                    if second_deriv < -self.tolerance:
                        return False
                except:
                    # If we can't compute second derivative, skip this point
                    continue
                    
            return True
            
        except Exception as e:
            print(f"Analytical method failed: {e}")
            return self.is_convex_numerical(func, domain, num_points)
    
    def is_convex_numerical(self, func: Callable, domain: Tuple[float, float], 
                          num_points: int = 1000) -> bool:
        """
        Check convexity using numerical methods (Jensen's inequality).
        
        Args:
            func: Function to test
            domain: Tuple of (min, max) values for the domain
            num_points: Number of random points to test
            
        Returns:
            bool: True if function appears convex, False otherwise
        """
        try:
            # Generate random points in the domain
            x_min, x_max = domain
            test_points = np.random.uniform(x_min, x_max, (num_points, 2))
            
            violations = 0
            total_tests = 0
            
            for x1, x2 in test_points:
                # Test Jensen's inequality: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
                lambda_val = np.random.uniform(0, 1)
                
                try:
                    # Left side: f(λx + (1-λ)y)
                    left_side = func(lambda_val * x1 + (1 - lambda_val) * x2)
                    
                    # Right side: λf(x) + (1-λ)f(y)
                    right_side = lambda_val * func(x1) + (1 - lambda_val) * func(x2)
                    
                    # Check if Jensen's inequality is violated
                    if left_side > right_side + self.tolerance:
                        violations += 1
                    
                    total_tests += 1
                    
                except (ValueError, ZeroDivisionError):
                    # Skip points where function is undefined
                    continue
            
            # Function is convex if violations are rare (less than 1% of tests)
            violation_rate = violations / max(total_tests, 1)
            return violation_rate < 0.01
            
        except Exception as e:
            print(f"Numerical method failed: {e}")
            return False
    
    def visualize_function(self, func: Callable, domain: Tuple[float, float], 
                         title: str = "Function Plot", num_points: int = 1000):
        """
        Plot the function and highlight convex/non-convex regions.
        
        Args:
            func: Function to visualize
            domain: Tuple of (min, max) values for the domain
            title: Title for the plot
            num_points: Number of points to plot
        """
        try:
            x_min, x_max = domain
            x = np.linspace(x_min, x_max, num_points)
            y = [func(xi) for xi in x]
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', linewidth=2, label='Function')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Test convexity and add annotation
            is_convex = self.is_convex_numerical(func, domain)
            convexity_text = "Convex" if is_convex else "Non-convex"
            plt.text(0.02, 0.98, f"Status: {convexity_text}", 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    def test_common_functions(self):
        """
        Test a collection of common convex and non-convex functions.
        """
        print("Testing Common Functions for Convexity")
        print("=" * 50)
        
        # Define test functions
        test_functions = {
            # Convex functions
            "x²": lambda x: x**2,
            "|x|": lambda x: abs(x),
            "e^x": lambda x: np.exp(x),
            "-log(x)": lambda x: -np.log(x) if x > 0 else np.inf,
            "x⁴": lambda x: x**4,
            
            # Non-convex functions
            "-x²": lambda x: -x**2,
            "sin(x)": lambda x: np.sin(x),
            "x³": lambda x: x**3,
            "cos(x)": lambda x: np.cos(x),
        }
        
        # Define domains for each function
        domains = {
            "x²": (-2, 2),
            "|x|": (-2, 2),
            "e^x": (-2, 2),
            "-log(x)": (0.1, 2),
            "x⁴": (-2, 2),
            "-x²": (-2, 2),
            "sin(x)": (-np.pi, np.pi),
            "x³": (-2, 2),
            "cos(x)": (-np.pi, np.pi),
        }
        
        results = {}
        
        for name, func in test_functions.items():
            domain = domains[name]
            
            # Test both methods
            analytical_result = self.is_convex_analytical(func, domain)
            numerical_result = self.is_convex_numerical(func, domain)
            
            results[name] = {
                'analytical': analytical_result,
                'numerical': numerical_result,
                'domain': domain
            }
            
            print(f"{name:12} | Analytical: {analytical_result:5} | Numerical: {numerical_result:5}")
        
        return results
    
    def analyze_function(self, func: Callable, domain: Tuple[float, float], 
                        name: str = "Function"):
        """
        Comprehensive analysis of a function's convexity.
        
        Args:
            func: Function to analyze
            domain: Domain to analyze
            name: Name of the function for display
        """
        print(f"\nAnalyzing {name}")
        print("=" * 30)
        
        # Test convexity
        analytical_result = self.is_convex_analytical(func, domain)
        numerical_result = self.is_convex_numerical(func, domain)
        
        print(f"Analytical test: {'Convex' if analytical_result else 'Non-convex'}")
        print(f"Numerical test:  {'Convex' if numerical_result else 'Non-convex'}")
        
        # Visualize
        self.visualize_function(func, domain, f"{name} - Convexity Analysis")
        
        return analytical_result and numerical_result


def main():
    """
    Main function to demonstrate the ConvexFunctionChecker.
    """
    print("Convex Function Checker - Lecture 1 Challenge")
    print("=" * 50)
    
    # Initialize checker
    checker = ConvexFunctionChecker()
    
    # Test common functions
    results = checker.test_common_functions()
    
    # Detailed analysis of a few interesting functions
    print("\nDetailed Analysis")
    print("=" * 20)
    
    # Test x² (clearly convex)
    checker.analyze_function(lambda x: x**2, (-2, 2), "x²")
    
    # Test sin(x) (clearly non-convex)
    checker.analyze_function(lambda x: np.sin(x), (-np.pi, np.pi), "sin(x)")
    
    # Test a more complex function
    complex_func = lambda x: x**4 - 2*x**2 + 1
    checker.analyze_function(complex_func, (-2, 2), "x⁴ - 2x² + 1")
    
    print("\nChallenge completed! 🎉")
    print("You've successfully implemented a convex function checker!")


if __name__ == "__main__":
    main()
