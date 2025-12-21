"""
Feature Scaling - High-Level Overview with Examples
"""

import numpy as np

def min_max_scale(data):
    """Min-Max scaling: scales to [0, 1]"""
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def z_score_standardize(data):
    """Z-Score standardization: mean=0, std=1"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return [0.0] * len(data)
    return [(x - mean_val) / std_val for x in data]

print("=" * 70)
print("FEATURE SCALING - HIGH-LEVEL OVERVIEW")
print("=" * 70)

# Example 1: The Problem
print("\nExample 1: Why We Need Feature Scaling")
print("-" * 70)

print("House Price Prediction Features:")
house_sizes = [1000, 2000, 3000, 4000, 5000]  # sqft
bedrooms = [1, 2, 3, 4, 5]  # count
ages = [0, 10, 20, 30, 40]  # years

print(f"  House sizes: {house_sizes} (range: 1000-5000)")
print(f"  Bedrooms:    {bedrooms} (range: 1-5)")
print(f"  Ages:        {ages} (range: 0-40)")

print("\nProblem: Different scales!")
print("  - House size values are 100-1000x larger")
print("  - Gradient descent will focus on size, ignore bedrooms/age")
print("  - Distance calculations dominated by size")

# Example 2: Min-Max Scaling
print("\n\nExample 2: Min-Max Scaling (Normalization)")
print("-" * 70)

print("Formula: (x - min) / (max - min)")
print("Result: All values between 0 and 1")

sizes_scaled = min_max_scale(house_sizes)
beds_scaled = min_max_scale(bedrooms)
ages_scaled = min_max_scale(ages)

print(f"\nOriginal house sizes: {house_sizes}")
print(f"Min-Max scaled:       {[round(s, 2) for s in sizes_scaled]}")
print("  All values now between 0 and 1!")

print(f"\nOriginal bedrooms:    {bedrooms}")
print(f"Min-Max scaled:       {[round(s, 2) for s in beds_scaled]}")
print("  Same scale as house sizes now!")

print(f"\nOriginal ages:        {ages}")
print(f"Min-Max scaled:       {[round(s, 2) for s in ages_scaled]}")
print("  All features on same scale!")

# Example 3: Z-Score Standardization
print("\n\nExample 3: Z-Score Standardization")
print("-" * 70)

print("Formula: (x - mean) / std")
print("Result: Mean = 0, Standard deviation = 1")

sizes_std = z_score_standardize(house_sizes)
beds_std = z_score_standardize(bedrooms)
ages_std = z_score_standardize(ages)

print(f"\nOriginal house sizes: {house_sizes}")
print(f"Mean: {np.mean(house_sizes):.1f}, Std: {np.std(house_sizes):.1f}")
print(f"Standardized:         {[round(float(s), 2) for s in sizes_std]}")
print("  Mean ~ 0, Std ~ 1")

print(f"\nOriginal bedrooms:    {bedrooms}")
print(f"Mean: {np.mean(bedrooms):.1f}, Std: {np.std(bedrooms):.1f}")
print(f"Standardized:    {[round(s, 2) for s in beds_std]}")
print("  Mean ~ 0, Std ~ 1")

# Example 4: Why It Matters for Gradient Descent
print("\n\nExample 4: Impact on Gradient Descent")
print("-" * 70)

print("Without Scaling:")
print("  Feature 1 (size): gradient = 5000 (huge!)")
print("  Feature 2 (bedrooms): gradient = 0.5 (tiny!)")
print("  Problem: Algorithm focuses on size, ignores bedrooms")
print("  Result: Slow convergence, poor model")

print("\nWith Scaling:")
print("  Feature 1 (size): gradient = 0.5 (normalized)")
print("  Feature 2 (bedrooms): gradient = 0.5 (normalized)")
print("  Benefit: All features contribute equally")
print("  Result: Fast convergence, better model")

# Example 5: Distance-Based Algorithms
print("\n\nExample 5: Impact on Distance Calculations")
print("-" * 70)

print("K-Nearest Neighbors (KNN) uses distances:")
print("\nWithout Scaling:")
house1 = [1000, 2]  # size, bedrooms
house2 = [2000, 3]
distance = np.sqrt((2000-1000)**2 + (3-2)**2)
print(f"  Distance = sqrt((2000-1000)² + (3-2)²)")
print(f"  Distance = {distance:.1f}")
print("  Problem: Dominated by size (1000 vs 1)")

print("\nWith Scaling:")
house1_scaled = [0.0, 0.25]  # scaled
house2_scaled = [0.25, 0.5]
distance_scaled = np.sqrt((0.25-0.0)**2 + (0.5-0.25)**2)
print(f"  Distance = sqrt((0.25-0)² + (0.5-0.25)²)")
print(f"  Distance = {distance_scaled:.2f}")
print("  Benefit: Both features contribute fairly!")

# Example 6: When You DON'T Need Scaling
print("\n\nExample 6: When Scaling Isn't Needed")
print("-" * 70)

print("Tree-Based Algorithms:")
print("  - Random Forest")
print("  - XGBoost")
print("  - Decision Trees")
print("  Why: They use splits, not distances")
print("  Result: Scaling doesn't help (but doesn't hurt)")

print("\nAlready Normalized Data:")
print("  - Image pixels: 0-255 (same scale)")
print("  - One-hot encoded: 0 or 1 (same scale)")
print("  - Already scaled features")
print("  Result: No scaling needed!")

# Example 7: Common Mistakes
print("\n\nExample 7: Common Mistakes to Avoid")
print("-" * 70)

print("Mistake 1: Scaling test data with test statistics")
print("  Wrong: test_scaled = (test - test.mean()) / test.std()")
print("  Correct: test_scaled = (test - train.mean()) / train.std()")
print("  Why: Must use training statistics!")

print("\nMistake 2: Scaling categorical features")
print("  Wrong: Scaling one-hot encoded [0, 1, 0]")
print("  Correct: Only scale numerical features")
print("  Why: Categorical features are already on same scale")

print("\nMistake 3: Scaling after train/test split")
print("  Wrong: Scale all data, then split")
print("  Correct: Split first, then scale using training stats")
print("  Why: Test data should be scaled same as training")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: FEATURE SCALING")
print("=" * 70)
print("\nWhat: Normalize features to similar scales")
print("\nWhy:")
print("  1. Gradient descent works better")
print("  2. Distance-based algorithms need it")
print("  3. Neural networks train faster")
print("\nMethods:")
print("  - Min-Max: [0, 1] range")
print("  - Z-Score: Mean=0, Std=1")
print("\nWhen:")
print("  - Always: Gradient descent, KNN, Neural networks")
print("  - Never: Tree-based algorithms")
print("\nKey Point: Use training statistics for test data!")
print("=" * 70)

