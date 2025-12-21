"""
GRADIENT DESCENT - THE FOUNDATION OF MACHINE LEARNING
======================================================

WHAT IS GRADIENT DESCENT?
--------------------------
Gradient Descent is an optimization algorithm that finds the minimum of a function
by taking steps in the direction of steepest descent (negative gradient).

Think of it like this:
- You're blindfolded on a hill
- You want to get to the bottom (minimum)
- You feel the slope with your feet (gradient)
- You take steps downhill (update parameters)
- Repeat until you reach the bottom (convergence)

WHY IS IT SO IMPORTANT?
-----------------------
- Foundation for training neural networks
- Used in linear regression, logistic regression, SVMs
- Core of backpropagation
- Used in almost every ML algorithm
- Concept appears in all CMU MSML courses

THE INTUITION:
--------------
1. Start with random guess (initial parameters)
2. Calculate how wrong you are (loss/error)
3. Calculate gradient (direction of steepest increase in error)
4. Move OPPOSITE to gradient (decrease error)
5. Repeat until error is minimized

KEY CONCEPTS:
-------------
- Gradient: Direction of steepest increase (points uphill)
- Learning Rate (alpha): How big steps to take
- Too small: Slow convergence
- Too large: Might overshoot minimum
- Just right: Fast convergence to minimum
"""

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))

    # Reshape y to be a column vector if it is not already
    y = y.reshape(m, 1)
    
    for i in range(iterations): 
        h = X @ theta
        error = h - y 
        gradient = (X.T @ error) / m  # study gradient meaning a bit more 
        theta = theta - alpha * gradient  # Update theta

    # Round the coefficients to four decimal places
    theta_rounded = np.round(theta, 4)

    return theta_rounded.flatten()  # Flatten to 1D array for the output, optional due to pre-reshaped above


# ============================================================================
# CONCEPTUAL DEEP DIVE: Understanding Gradient Descent
# ============================================================================

def explain_gradient_descent_conceptually():
    """
    Deep dive into understanding Gradient Descent conceptually.
    """
    print("=" * 70)
    print("GRADIENT DESCENT - CONCEPTUAL DEEP DIVE")
    print("=" * 70)
    
    print("\n1. THE HILL CLIMBING ANALOGY")
    print("-" * 70)
    print("Imagine you're blindfolded on a hill and want to reach the bottom:")
    print("  - You can't see where the bottom is")
    print("  - But you can feel the slope with your feet")
    print("  - The slope tells you which direction is downhill")
    print("  - You take steps in that direction")
    print("  - Eventually you reach the bottom!")
    print("\nGradient Descent works the same way:")
    print("  - 'Hill' = our error/loss function")
    print("  - 'Bottom' = minimum error (best parameters)")
    print("  - 'Slope' = gradient (direction of steepest increase)")
    print("  - 'Steps' = parameter updates")
    
    print("\n\n2. WHAT IS A GRADIENT?")
    print("-" * 70)
    print("Gradient = Direction of steepest INCREASE in error")
    print("\nKey insight: We want to DECREASE error, so we move OPPOSITE to gradient!")
    print("\nMathematical definition:")
    print("  Gradient = partial derivative of error with respect to each parameter")
    print("  For linear regression: gradient = (X.T @ error) / m")
    print("\nWhat it tells us:")
    print("  - If gradient is positive: increasing parameter increases error")
    print("  - If gradient is negative: increasing parameter decreases error")
    print("  - Magnitude: how steep the slope is")
    
    print("\n\n3. THE UPDATE RULE")
    print("-" * 70)
    print("theta = theta - alpha * gradient")
    print("\nBreaking it down:")
    print("  theta: current parameter values")
    print("  alpha: learning rate (step size)")
    print("  gradient: direction of steepest increase")
    print("  -alpha * gradient: move OPPOSITE to gradient (downhill)")
    print("\nWhy subtract?")
    print("  - Gradient points uphill (direction of increasing error)")
    print("  - We want to go downhill (decrease error)")
    print("  - So we subtract: move opposite direction!")
    
    print("\n\n4. LEARNING RATE (ALPHA) - THE CRITICAL PARAMETER")
    print("-" * 70)
    print("Learning rate controls step size:")
    print("\nToo Small (e.g., alpha = 0.0001):")
    print("  - Takes tiny steps")
    print("  - Very slow convergence")
    print("  - Might get stuck in local minima")
    print("  - But safe - won't overshoot")
    
    print("\nToo Large (e.g., alpha = 10):")
    print("  - Takes huge steps")
    print("  - Might overshoot minimum")
    print("  - Could diverge (error increases)")
    print("  - Unstable")
    
    print("\nJust Right (e.g., alpha = 0.01):")
    print("  - Balanced step size")
    print("  - Fast convergence")
    print("  - Stable")
    print("  - Reaches minimum efficiently")
    
    print("\n\n5. HOW IT WORKS STEP-BY-STEP")
    print("-" * 70)
    print("Step 1: Initialize parameters randomly")
    print("  theta = [0, 0]  (or random values)")
    
    print("\nStep 2: Make predictions")
    print("  predictions = X @ theta")
    print("  Example: X = [[1, 2], [1, 3]], theta = [1, 1]")
    print("  predictions = [1×1 + 2×1, 1×1 + 3×1] = [3, 4]")
    
    print("\nStep 3: Calculate error")
    print("  error = predictions - actual_values")
    print("  Example: predictions = [3, 4], actual = [5, 7]")
    print("  error = [3-5, 4-7] = [-2, -3]")
    
    print("\nStep 4: Calculate gradient")
    print("  gradient = (X.T @ error) / m")
    print("  This tells us: 'How should we change theta to reduce error?'")
    
    print("\nStep 5: Update parameters")
    print("  theta = theta - alpha * gradient")
    print("  Move in direction that decreases error")
    
    print("\nStep 6: Repeat")
    print("  Go back to Step 2, repeat until convergence")
    
    print("\n\n6. VISUALIZING GRADIENT DESCENT")
    print("-" * 70)
    print("Imagine a 2D error surface (like a bowl):")
    print("""
         Error
           |
        High|     * (start here)
           |    / \\
           |   /   \\
           |  /     \\
        Low|_/_______\\_ (minimum here)
           |
           |__|__|__|__|__ Parameters
    """)
    print("\nGradient Descent:")
    print("  1. Start at random point (high error)")
    print("  2. Calculate gradient (direction uphill)")
    print("  3. Move opposite (downhill)")
    print("  4. Repeat until bottom (minimum error)")
    
    print("\n\n7. CONVERGENCE - WHEN DO WE STOP?")
    print("-" * 70)
    print("We stop when:")
    print("  - Error stops decreasing significantly")
    print("  - Gradient becomes very small (flat surface)")
    print("  - We've done enough iterations")
    print("  - Parameters stop changing much")
    print("\nIn practice:")
    print("  - Run for fixed number of iterations")
    print("  - Or stop when error change < threshold")
    print("  - Or stop when gradient magnitude < threshold")
    
    print("\n\n8. WHY IT WORKS - THE MATHEMATICS")
    print("-" * 70)
    print("For linear regression, we minimize Mean Squared Error (MSE):")
    print("  MSE = (1/m) * sum((predictions - actual)^2)")
    print("\nTaking derivative with respect to theta:")
    print("  gradient = (2/m) * X.T @ (predictions - actual)")
    print("  gradient = (2/m) * X.T @ error")
    print("\nThe gradient points in direction of steepest increase.")
    print("Moving opposite (theta - alpha * gradient) decreases error!")
    
    print("\n\n9. COMMON VARIANTS")
    print("-" * 70)
    print("Batch Gradient Descent (what we implemented):")
    print("  - Uses ALL training examples to compute gradient")
    print("  - Stable, but slow for large datasets")
    
    print("\nStochastic Gradient Descent (SGD):")
    print("  - Uses ONE random example per iteration")
    print("  - Fast, but noisy (less stable)")
    
    print("\nMini-Batch Gradient Descent:")
    print("  - Uses small batch (e.g., 32 examples)")
    print("  - Balance between speed and stability")
    print("  - Most common in practice!")
    
    print("\n\n10. REAL-WORLD ANALOGY")
    print("-" * 70)
    print("Think of training a dog:")
    print("  - Dog tries trick (makes prediction)")
    print("  - You see how wrong it was (calculate error)")
    print("  - You give correction (gradient)")
    print("  - Dog adjusts (updates parameters)")
    print("  - Repeat until dog learns trick (convergence)")
    print("\nGradient Descent is like this, but mathematical!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. Gradient points uphill (direction of increasing error)")
    print("2. We move opposite (downhill) to decrease error")
    print("3. Learning rate controls step size - critical parameter!")
    print("4. Iterative process - keep updating until convergence")
    print("5. Foundation for ALL neural network training")
    print("=" * 70)


# Example usage with sample data
if __name__ == "__main__":
    # First, explain the concept
    explain_gradient_descent_conceptually()
    
    print("\n\n" + "=" * 70)
    print("NOW LET'S SEE IT IN ACTION WITH CODE EXAMPLES")
    print("=" * 70)
    
    print("\n" + "=" * 60)
    print("Linear Regression with Gradient Descent - Example")
    print("=" * 60)
    
    # Example 1: Simple 2D data (1 feature + bias)
    # We want to predict y = 2 + 3*x (so theta should be [2, 3])
    print("\nExample 1: Simple Linear Relationship")
    print("-" * 60)
    
    # Create sample data: y = 2 + 3*x (approximately)
    X_simple = np.array([
        [1, 1],  # [bias=1, feature=1]
        [1, 2],  # [bias=1, feature=2]
        [1, 3],  # [bias=1, feature=3]
        [1, 4],  # [bias=1, feature=4]
        [1, 5]   # [bias=1, feature=5]
    ])
    
    y_simple = np.array([5, 8, 11, 14, 17])  # y = 2 + 3*x
    
    print(f"Input features X:\n{X_simple}")
    print(f"\nTarget values y: {y_simple}")
    print(f"\nTrue relationship: y = 2 + 3*x")
    
    # Run gradient descent
    theta_simple = linear_regression_gradient_descent(X_simple, y_simple, alpha=0.01, iterations=1000)
    
    print(f"\nLearned coefficients (theta): {theta_simple}")
    print(f"   Expected: [2.0, 3.0] (intercept=2, slope=3)")
    print(f"   Got:      [{theta_simple[0]:.4f}, {theta_simple[1]:.4f}]")
    
    # Make predictions
    predictions = X_simple @ theta_simple.reshape(-1, 1)
    print(f"\nPredictions:")
    for i in range(len(y_simple)):
        print(f"   x={X_simple[i,1]:.1f} -> predicted={predictions[i,0]:.2f}, actual={y_simple[i]:.2f}")
    
    # Example 2: Multiple features
    print("\n\nExample 2: Multiple Features (House Price Prediction)")
    print("-" * 60)
    
    # Features: [bias, size (sqft), bedrooms]
    X_house = np.array([
        [1, 1000, 2],  # Small house, 2 bedrooms
        [1, 1500, 3],  # Medium house, 3 bedrooms
        [1, 2000, 3],  # Large house, 3 bedrooms
        [1, 2500, 4],  # Very large house, 4 bedrooms
        [1, 3000, 4]   # Huge house, 4 bedrooms
    ])
    
    # Prices (in thousands): roughly price = 50 + 0.1*size + 20*bedrooms
    y_house = np.array([90, 140, 190, 240, 290])
    
    print(f"Features: [bias, size(sqft), bedrooms]")
    print(f"X:\n{X_house}")
    print(f"\nPrices (in $1000s): {y_house}")
    
    theta_house = linear_regression_gradient_descent(X_house, y_house, alpha=0.0001, iterations=2000)
    
    print(f"\nLearned coefficients: {theta_house}")
    print(f"   Interpretation:")
    print(f"   - Base price (intercept): ${theta_house[0]:.2f}k")
    print(f"   - Price per sqft: ${theta_house[1]:.4f}k")
    print(f"   - Price per bedroom: ${theta_house[2]:.2f}k")
    
    # Example 3: Show convergence over iterations
    print("\n\nExample 3: Watching Convergence")
    print("-" * 60)
    
    X_conv = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    y_conv = np.array([3, 5, 7, 9])  # y = 1 + 2*x
    
    print("Tracking theta values over iterations:")
    print("Iteration |  Theta[0] (intercept) |  Theta[1] (slope)")
    print("-" * 60)
    
    # Modified version to show progress
    m, n = X_conv.shape
    theta_conv = np.zeros((n, 1))
    y_conv = y_conv.reshape(m, 1)
    alpha = 0.1
    
    for i in range(0, 20, 2):  # Show every 2 iterations
        if i > 0:
            for _ in range(2):
                h = X_conv @ theta_conv
                error = h - y_conv
                gradient = (X_conv.T @ error) / m
                theta_conv = theta_conv - alpha * gradient
        print(f"   {i:3d}    |      {theta_conv[0,0]:8.4f}        |    {theta_conv[1,0]:8.4f}")
    
    print(f"\nFinal theta: [{theta_conv[0,0]:.4f}, {theta_conv[1,0]:.4f}]")
    print(f"   Expected: [1.0, 2.0]")
    
    # Example 4: Student Exam Scores (Study Hours vs Score)
    print("\n\nExample 4: Predicting Exam Scores from Study Hours")
    print("-" * 60)
    
    # Features: [bias, study_hours]
    X_study = np.array([
        [1, 5],   # 5 hours of study
        [1, 10],  # 10 hours
        [1, 15],  # 15 hours
        [1, 20],  # 20 hours
        [1, 25],  # 25 hours
        [1, 30]   # 30 hours
    ])
    
    # Exam scores: roughly score = 50 + 2*study_hours (with some variation)
    y_scores = np.array([60, 70, 80, 90, 100, 110])
    
    print("Study Hours vs Exam Scores:")
    for i in range(len(X_study)):
        print(f"   {X_study[i,1]:2d} hours -> Score: {y_scores[i]}")
    
    print(f"\nRunning gradient descent...")
    theta_study = linear_regression_gradient_descent(X_study, y_scores, alpha=0.001, iterations=2000)
    
    print(f"\nLearned model: Score = {theta_study[0]:.2f} + {theta_study[1]:.2f} * study_hours")
    print(f"   Interpretation:")
    print(f"   - Base score (no studying): {theta_study[0]:.2f} points")
    print(f"   - Points per hour of studying: {theta_study[1]:.2f}")
    
    # Make predictions for new students
    print(f"\nPredictions for new students:")
    new_students = [8, 12, 18, 22]
    for hours in new_students:
        predicted_score = theta_study[0] + theta_study[1] * hours
        print(f"   {hours:2d} hours -> Predicted score: {predicted_score:.1f}")
    
    # Example 5: Temperature vs Ice Cream Sales
    print("\n\nExample 5: Temperature vs Ice Cream Sales")
    print("-" * 60)
    
    # Features: [bias, temperature_F]
    X_temp = np.array([
        [1, 60],   # 60°F
        [1, 65],   # 65°F
        [1, 70],   # 70°F
        [1, 75],   # 75°F
        [1, 80],   # 80°F
        [1, 85],   # 85°F
        [1, 90]    # 90°F
    ])
    
    # Sales in units: roughly sales = -100 + 5*temperature
    y_sales = np.array([200, 225, 250, 275, 300, 325, 350])
    
    print("Temperature vs Ice Cream Sales:")
    for i in range(len(X_temp)):
        print(f"   {X_temp[i,1]:2d}°F -> {y_sales[i]:3d} units sold")
    
    theta_sales = linear_regression_gradient_descent(X_temp, y_sales, alpha=0.0001, iterations=3000)
    
    print(f"\nLearned model: Sales = {theta_sales[0]:.2f} + {theta_sales[1]:.2f} * temperature")
    print(f"   Interpretation:")
    print(f"   - Base sales (at 0°F): {theta_sales[0]:.2f} units")
    print(f"   - Additional sales per degree: {theta_sales[1]:.2f} units")
    
    # Predict for a hot day
    hot_day_temp = 95
    predicted_sales = theta_sales[0] + theta_sales[1] * hot_day_temp
    print(f"\n   Prediction for {hot_day_temp}°F day: {predicted_sales:.0f} units")
    
    print("\n" + "=" * 60)
    print("Gradient Descent Complete!")
    print("=" * 60) 
