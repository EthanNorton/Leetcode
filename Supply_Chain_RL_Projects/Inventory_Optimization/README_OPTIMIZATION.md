# Optimization Setup Guide

## Quick Start

The notebook `grainger_eda_optimization.ipynb` has been created with:

### Part 1: EDA (Cells 1-6)
- Product portfolio analysis
- Promotion impact visualization
- Lead time patterns
- Demand correlation analysis

### Part 2: PuLP Optimization (Cells 7-10)
- Truckload optimization model
- Decision variables for each SKU
- Weight and volume constraints
- Objective: Maximize fulfillment

### Part 3: Gurobi Setup (Cells 11-13)
- Alternative solver (if available)
- Same problem formulated for Gurobi
- Performance comparison

### Part 4: Sensitivity Analysis (Cells 14-18)
- Test different discount levels (10%, 15%, 20%, 25%)
- Vary truck capacity (±20%)
- Change service level targets
- Analyze lead time impact

## How to Run

```bash
# Install required packages
pip install pulp pandas numpy matplotlib seaborn scipy jupyter

# Optional: Install Gurobi (requires license)
pip install gurobipy

# Start Jupyter
jupyter notebook grainger_eda_optimization.ipynb
```

## Optimization Problems Included

### 1. Truckload Optimization (PuLP)
```python
# Decision: How many units of each SKU to order
# Maximize: Total fulfillment rate
# Subject to: Weight limit, Volume limit
```

### 2. Promotion Planning (PuLP)
```python
# Decision: Order quantity for promoted items
# Maximize: Profit - stockout cost
# Subject to: Budget constraint, service level
```

### 3. Multi-Supplier Selection (Gurobi)
```python
# Decision: Which supplier to use for each SKU
# Minimize: Total cost (purchase + transportation)
# Subject to: Lead time requirements, capacity
```

## Sensitivity Analysis Framework

The notebook includes ready-to-run sensitivity analysis for:

1. **Promotion Discount Sensitivity**
   - Test 10%, 15%, 20%, 25% discounts
   - Measure demand lift vs profitability
   - Find optimal discount level

2. **Capacity Sensitivity**
   - Vary truck capacity ±20%
   - Plot optimal order quantities
   - Identify bottlenecks

3. **Lead Time Sensitivity**
   - Simulate lead time delays
   - Calculate safety stock impact
   - Risk analysis

4. **Cost Sensitivity**
   - Change unit costs ±10%
   - Update profit margins
   - Reoptimize decisions

## Next Steps

1. Open the notebook
2. Run all cells to see baseline results
3. Modify sensitivity parameters
4. Build your own scenarios
5. Export results for presentation

Happy optimizing! 🚀
