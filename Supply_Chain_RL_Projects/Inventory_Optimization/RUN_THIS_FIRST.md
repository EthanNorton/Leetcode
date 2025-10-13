# 🚀 Ready-to-Run Optimization Framework

## What You Have Now

✅ **Complete datasets** (4 CSV files with realistic supply chain data)  
✅ **Working optimization code** (PuLP with Gurobi structure ready)  
✅ **Sensitivity analysis** (capacity, cost, promotion impact)  
✅ **Visualizations** (auto-generated charts and insights)

## Quick Start (2 Commands!)

### Option 1: Run Complete Analysis (Python Script)
```bash
cd Supply_Chain_RL_Projects/Inventory_Optimization
python optimization_framework.py
```

**You'll get:**
- ✅ Optimal truckload order (which SKUs, how many)
- ✅ Capacity sensitivity (how truck size affects profit)
- ✅ Cost sensitivity (how price changes affect decisions)
- ✅ Promotion analysis (optimal discount levels)
- ✅ 3 visualization files saved automatically

### Option 2: Interactive Notebook
```bash
jupyter notebook grainger_eda_optimization.ipynb
```

**Run cells to:**
- Explore data visually
- Modify optimization parameters
- Test different scenarios
- Export custom results

## What the Code Does

### 1. Truckload Optimization (PuLP)
```python
# Maximize: Total profit
# Subject to:
#   - Weight ≤ 20,000 lbs
#   - Volume ≤ 1,200 cu ft
#   - Order qty ≤ Needed qty
```

**Output Example:**
```
Optimal Order:
  SKU001: 500 units
  SKU003: 250 units
  SKU005: 400 units
  
Net Profit: $18,450
Fulfillment Rate: 87%
Weight Utilization: 71%
```

### 2. Sensitivity Analysis

**A) Capacity Sensitivity**
- Tests truck sizes from 70% to 130% of base
- Shows profit vs capacity curve
- Identifies diminishing returns point

**B) Cost Sensitivity**
- Varies unit costs ±10%
- Measures profit impact
- Helps with pricing decisions

**C) Promotion Sensitivity**
- Analyzes 10%, 15%, 20%, 25% discounts
- Measures demand lift
- Tracks service level impact

## Files You'll Get

After running `optimization_framework.py`:

1. **sensitivity_capacity.png** - Capacity impact charts
2. **sensitivity_cost.png** - Cost change analysis
3. **sensitivity_promotion.png** - Promotion effectiveness

## Gurobi Setup (Optional - Advanced)

If you want to use Gurobi (faster for large problems):

1. Get free academic license at gurobi.com
2. Install: `pip install gurobipy`
3. Uncomment Gurobi function in `optimization_framework.py` (line 107)

The code works with PuLP by default!

## Customize Your Analysis

### Change Needed Quantities
```python
# In optimization_framework.py, line 270
needed_skus = {
    'SKU001': 500,  # Change these numbers
    'SKU003': 300,
    'SKU005': 400,
    # Add more SKUs...
}
```

### Adjust Sensitivity Ranges
```python
# Line 170 - Capacity range
capacity_range = np.linspace(0.5, 1.5, 11)  # 50% to 150%

# Line 210 - Cost range  
cost_range = np.linspace(0.8, 1.2, 9)  # ±20%
```

### Add New Constraints
```python
# In optimize_truckload_pulp function
# Add after line 55:
prob += lpSum([order_qty[sku] for sku in order_qty.keys()]) >= 1000, "Min_Total_Units"
```

## Understanding the Output

### Optimization Status
- **"Optimal"** = Found best solution ✓
- **"Infeasible"** = Constraints too tight ✗
- **"Unbounded"** = Need more constraints ✗

### Key Metrics
- **Net Profit** = Gross profit - Transportation cost
- **Fulfillment Rate** = Orders filled / Orders needed
- **Utilization** = Used capacity / Max capacity

### Sensitivity Insights
Look for:
- 📈 **Steep slopes** = High sensitivity (small changes, big impact)
- 📊 **Flat regions** = Low sensitivity (robust to changes)
- 🎯 **Breakpoints** = Where strategy should change

## Troubleshooting

**Error: "No module named pulp"**
```bash
pip install pulp
```

**Error: "FileNotFoundError"**
```bash
# Make sure you're in the right directory
cd Supply_Chain_RL_Projects/Inventory_Optimization
ls data/  # Should see 4 CSV files
```

**Plots don't show**
```python
# Add at end of script
plt.show()
```

## Interview Prep

Use this to answer:

**Q: "How do you handle optimization under constraints?"**  
A: "I built a truckload optimization using linear programming with PuLP. Maximized profit subject to weight and volume constraints..."

**Q: "Describe sensitivity analysis"**  
A: "I tested how capacity, costs, and promotions affect optimal decisions. Found that ±10% cost change impacts profit by $X..."

**Q: "Gurobi vs open-source solvers?"**  
A: "I've used both. PuLP works for most problems. Gurobi is 10-100x faster for large-scale optimization, worth it for real-time decisions..."

## Next Steps

1. ✅ Run `python optimization_framework.py`
2. ✅ Review the 3 generated charts
3. ✅ Modify parameters and re-run
4. ✅ Add to your portfolio with screenshots
5. ✅ Practice explaining to interviewers

**You're ready to optimize! 🎯**

---

*This framework solves all 3 Grainger problems with sensitivity analysis - perfect for demonstrating advanced supply chain optimization skills!*

