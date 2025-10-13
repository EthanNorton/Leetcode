# Quick Start: Grainger-Style Supply Chain Problem

## 🚀 Run This in 5 Minutes

### Step 1: Navigate to the folder
```bash
cd "C:\Users\ethan\Downloads\classes\Leetcode\Supply_Chain_RL_Projects\Inventory_Optimization"
```

### Step 2: Run the complete analysis
```bash
python grainger_combined_problem.py
```

That's it! You'll see:
- ✅ Promotion impact analysis
- ✅ Lead time predictions with intervals
- ✅ Optimized truckload recommendations
- ✅ Visualizations saved to `grainger_analysis.png`

## 📊 What You Get

### The Datasets (Already Created!)
- `data/sku_master.csv` - 15 products with costs, weights, suppliers
- `data/weekly_demand_multi_sku.csv` - 80+ weeks of demand with promotions
- `data/supplier_leadtimes.csv` - 30 actual delivery records
- `data/truckload_constraints.csv` - Truck capacity limits

### The Problems (All Combined!)
1. **Promotion Impact**: Predict demand during 10-25% discounts
2. **Lead Time Forecasting**: 90% prediction intervals for delivery
3. **Truckload Optimization**: Maximize fulfillment within weight/volume limits

## 🎯 Why This Matters for Grainger Role

From the job posting, they want you to:
- ✅ "Build a model that predicts the impact promotions have on demand" → **Problem 1**
- ✅ "Forecast supplier purchase order lead times with prediction intervals" → **Problem 2**  
- ✅ "Create an optimization model that prioritizes what we buy from a supplier, while ensuring the total amount purchased fits within a truckload space constraint" → **Problem 3**

**This code does ALL THREE!**

## 💡 Understanding the Output

### Promotion Impact Table
```
Discount_%  Demand_Multiplier  Lift_%
    20          1.85           85.0
```
Means: 20% discount increases demand by 85%

### Lead Time Predictions
```
Supplier  SKU    Mean  90%_Lower  90%_Upper
SUP_A    SKU001   7.3    6.1       8.5
```
Means: 90% confident delivery in 6-9 days

### Truck Optimization
```
Total Weight: 14,250 lbs (71% of max)
Fulfillment Rate: 87%
Net Profit: $18,450
```
Means: Efficiently using truck while maximizing profit

## 🔧 Quick Modifications

### Change Exploration Rate
```python
# In grainger_combined_problem.py, line ~350
order = truck_bandit.generate_feasible_order('SUP_A', needed_skus, epsilon=0.3)  # More exploration
```

### Adjust Service Level
```python
# Line ~120
recommended_qty = promo_bandit.recommend_inventory(145, 20, service_level=0.98)  # Higher service
```

### Try Different Promotions
```python
# Add to the needed_skus dict (line ~360)
needed_skus = {
    'SKU001': 500,
    'SKU003': 300,
    'SKU005': 400,
    'SKU009': 350,
    'SKU011': 600,
    'SKU013': 400  # Add metal file set
}
```

## 📈 Next Steps

### Beginner:
1. Run the script as-is
2. Read the output and understand each section
3. Modify one parameter at a time
4. See how results change

### Intermediate:
1. Add a new SKU to the master file
2. Simulate more weeks of demand
3. Try different truck types
4. Implement UCB instead of epsilon-greedy

### Advanced:
1. Add multi-objective optimization (cost + service + risk)
2. Implement Thompson Sampling
3. Add dynamic pricing (RL chooses discount %)
4. Build web dashboard with Streamlit

## 🎓 Portfolio Impact

**Before**: "I know RL theory"  
**After**: "I built a multi-problem supply chain optimizer using bandits, handling real-world constraints like truck capacity and demand uncertainty"

Add to your resume:
- ✅ Multi-armed bandits for business optimization
- ✅ Prediction intervals for uncertainty quantification
- ✅ Constrained optimization with greedy algorithms
- ✅ Real-world supply chain problem solving

## 🆘 Troubleshooting

**Error: "No module named pandas"**
```bash
pip install pandas numpy matplotlib seaborn scipy
```

**Error: "FileNotFoundError: data/sku_master.csv"**
```bash
# Make sure you're in the right directory
cd Supply_Chain_RL_Projects/Inventory_Optimization
```

**Plot doesn't show**
```python
# Add this at the end of the script
plt.show()  # Should already be there
```

## 💼 Interview Prep

Use this project to answer:

**"Tell me about a complex problem you solved with ML"**
→ Talk about combining 3 supply chain problems into unified RL solution

**"How do you handle uncertainty in predictions?"**
→ Explain prediction intervals and safety stock calculations

**"Give an example of optimization under constraints"**
→ Walk through the truckload optimization algorithm

**"How would you deploy this in production?"**
→ Discuss API endpoints, monitoring, A/B testing framework

---

**You're ready! Run the script and start exploring.** 🎯

