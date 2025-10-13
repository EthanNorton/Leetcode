# Supply Chain Optimization with Reinforcement Learning

Real-world supply chain problems solved with RL - Perfect for roles like Grainger's Supply Chain Optimization Engineer!

## About

These notebooks demonstrate how RL solves actual supply chain challenges:
- Inventory optimization
- Demand forecasting under uncertainty  
- Multi-echelon distribution
- Promotion impact modeling
- Lead time prediction

## Skills You'll Build

### Technical Skills (from Grainger job posting):
✅ Forecasting and optimization
✅ Working with large datasets (millions of rows)
✅ Python, NumPy, Pandas
✅ Statistical modeling
✅ Data visualization
✅ Linear/integer programming concepts

### RL Skills:
✅ Multi-armed bandits for A/B testing
✅ Contextual bandits for personalization
✅ MDP formulation of business problems
✅ Q-learning for sequential decisions
✅ Policy gradient methods

## Project Structure

### 1. Inventory_Optimization/
**Business Problem**: Decide how much inventory to order to balance holding costs vs stockouts

Notebooks:
- `01_single_product_bandit.ipynb` - Learn optimal order quantity
- `02_multi_product_sku_management.ipynb` - Manage portfolio of SKUs
- `03_seasonal_demand_adaptation.ipynb` - Adapt to demand changes

### 2. Promotion_Planning/
**Business Problem**: Predict promotion impact and optimize inventory

Notebooks:
- `01_promotion_impact_prediction.ipynb` - Forecast demand during promotions
- `02_dynamic_pricing_rl.ipynb` - RL for pricing decisions

### 3. Distribution_Network/
**Business Problem**: Optimize where to place inventory across DCs

Notebooks:
- `01_two_echelon_network.ipynb` - Warehouse & retail decisions
- `02_transportation_optimization.ipynb` - Route and quantity optimization

### 4. Lead_Time_Forecasting/
**Business Problem**: Predict supplier lead times with uncertainty

Notebooks:
- `01_leadtime_prediction_intervals.ipynb` - Forecasting with confidence bounds
- `02_supplier_selection_rl.ipynb` - Choose best suppliers over time

## Datasets

All datasets are **realistic simulations** based on public domain data:
- 10,000+ SKUs
- Multiple distribution centers
- Seasonal patterns
- Real-world constraints

## How to Use

1. **Start with Project 1** (Inventory Optimization)
2. Each notebook is self-contained
3. Run cells in order
4. Modify parameters and see results change
5. Try the challenges at the end

## Prerequisites

```python
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Connection to RL Book

- **Chapter 1-2**: Inventory_Optimization notebooks
- **Chapter 3**: Distribution_Network notebooks  
- **Chapter 5-6**: Promotion_Planning notebooks

## Real Interview Questions

At the end of each notebook, we include:
- Interview-style questions from companies like Grainger
- Code challenges similar to take-home assignments
- Discussion questions about trade-offs

Let's build your supply chain RL portfolio! 🚀
