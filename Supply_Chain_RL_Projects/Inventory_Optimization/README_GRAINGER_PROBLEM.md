# Grainger-Style Combined Supply Chain Problem

This is a **realistic supply chain optimization problem** that combines all three scenarios from the Grainger job posting.

## 🎯 The Business Problem

You're a Supply Chain Optimization Engineer at a major MRO distributor (like Grainger). You need to:

1. **Predict promotion impact** and set inventory levels to maintain 95%+ service levels
2. **Forecast supplier lead times** with prediction intervals for safety stock
3. **Optimize truckload orders** to maximize fulfillment within weight/volume constraints

## 📊 The Datasets

### 1. `sku_master.csv` (15 SKUs)
Product catalog with:
- Unit costs and selling prices
- Physical dimensions (weight, volume)
- Supplier relationships
- Average lead times and variability
- Promotional eligibility

**Key Fields:**
- `unit_cost`, `selling_price` → Profit calculations
- `weight_lbs`, `volume_cuft` → Truck constraints
- `avg_lead_time_days`, `lead_time_std` → Delivery uncertainty

### 2. `weekly_demand_multi_sku.csv` (80+ records)
Historical demand with:
- Base demand patterns
- Promotion effects (discount %, actual lift)
- Stockout occurrences
- Service level achievement
- External factors (weather, competition)

**Key Fields:**
- `promotion_discount_pct` → Test different promotion levels
- `actual_demand` vs `base_demand` → Learn lift
- `stockout_qty` → Measure underordering cost
- `service_level_pct` → Track performance

### 3. `supplier_leadtimes.csv` (30 orders)
Actual delivery performance with:
- Expected vs actual delivery dates
- Lead time variance by supplier
- Weather and transportation impacts
- Rush order capabilities

**Key Fields:**
- `lead_time_days` → Actual delivery time
- `expected_lead_time` → Supplier promise
- `lead_time_variance` → Reliability metric
- `transportation_mode` → Air vs Truck

### 4. `truckload_constraints.csv`
Truck capacity limits:
- Weight and volume maximums
- Cost per load
- Minimum utilization requirements
- SKU limits per load

**Key Fields:**
- `max_weight_lbs`, `max_volume_cuft` → Hard constraints
- `cost_per_load` → Fixed transportation cost
- `min_utilization_pct` → Efficiency target

## 🚀 How to Run

### Option 1: Run the Complete Analysis
```bash
cd Supply_Chain_RL_Projects/Inventory_Optimization
python grainger_combined_problem.py
```

This will:
1. Load all 4 datasets
2. Train 3 different RL agents (one per problem)
3. Make predictions and recommendations
4. Generate visualizations
5. Save results to `grainger_analysis.png`

### Option 2: Work with Individual Problems

You can import and use each component separately:

```python
from grainger_combined_problem import PromotionImpactBandit

# Learn promotion effects
promo_bandit = PromotionImpactBandit()
promo_bandit.update(discount=20, base_demand=145, actual_demand=220)

# Get recommendation
qty = promo_bandit.recommend_inventory(base_demand=150, discount=20)
print(f"Order {qty} units for 20% promotion")
```

## 🧠 The RL Approach

### Problem 1: Promotion Impact (Multi-Armed Bandit)
- **Arms**: Different discount levels (0%, 10%, 15%, 20%, 25%)
- **Reward**: How well we predict demand (negative of error)
- **Learning**: Track demand multiplier for each discount level
- **Output**: Recommended inventory = predicted demand + safety stock

### Problem 2: Lead Time Forecasting (Contextual Bandit)
- **Context**: (Supplier, SKU) pairs
- **Arms**: Not used - this is pure prediction
- **Reward**: Accuracy of prediction interval
- **Learning**: Track mean and std dev for each supplier-SKU combo
- **Output**: 90% prediction interval for delivery time

### Problem 3: Truckload Optimization (Epsilon-Greedy)
- **Arms**: Different SKU combinations to load
- **Reward**: Fulfillment rate (% of needed demand met)
- **Learning**: Try random combos (explore) vs greedy priority (exploit)
- **Output**: Optimal order that fits in truck

## 📈 Expected Results

When you run the complete analysis, you should see:

**Promotion Impact:**
```
Discount_%  Demand_Multiplier  Observations  Lift_%
    10          1.28              3          28.0
    15          1.52              4          52.0
    20          1.85              3          85.0
    25          2.19              2         119.0
```

**Lead Time Predictions:**
```
Supplier  SKU      Mean_LeadTime  90%_Lower  90%_Upper
SUP_A     SKU001        7.3         6.1        8.5
SUP_A     SKU003       10.2         8.8       11.6
SUP_B     SKU002       15.1        13.2       17.0
```

**Truckload Optimization:**
```
Total Weight: 14,250 lbs (71% of max)
Total Volume: 890 cu ft (74% of max)
Fulfillment Rate: 87%
Net Profit: $18,450
```

## 🎓 Learning Objectives

After working through this problem, you'll understand:

1. **How to formulate business problems as RL problems**
   - Identify states, actions, rewards
   - Choose appropriate RL algorithm

2. **How to handle uncertainty in supply chains**
   - Demand variability
   - Lead time uncertainty
   - Constraint optimization

3. **How to build prediction intervals**
   - Not just point estimates
   - Account for variance and sample size

4. **How to optimize under constraints**
   - Weight and volume limits
   - Multi-objective optimization (cost vs service)

## 💼 Interview Preparation

This problem demonstrates skills for roles like:
- **Grainger**: Supply Chain Optimization Engineer
- **Amazon**: Supply Chain Scientist
- **Walmart**: Inventory Optimization Analyst
- **Target**: Supply Chain Data Scientist

### Sample Interview Questions:

**Q1**: "Your 20% promotion prediction says 220 units demand, but we only ordered 200 and sold out. What happened?"

**A1**: Could be:
- Our multiplier estimate is biased low (need more data)
- External factors we didn't account for (competitor stockout, weather)
- Should've added more safety stock given uncertainty
- **Action**: Update the bandit with actual demand, increase safety stock factor

**Q2**: "Supplier lead time variance increased 50%. How do you adjust?"

**A2**: 
- Prediction intervals will widen automatically
- May need to increase safety stock
- Consider switching to more reliable supplier
- Implement rush order strategy for critical SKUs

**Q3**: "The optimization fills 95% of truck volume but only 60% by weight. Is this optimal?"

**A3**:
- Depends on what's being shipped (light but bulky items)
- Should add value density to optimization objective
- Consider mixed supplier orders to balance utilization
- May need different truck types (volume vs weight optimized)

## 🔧 Modifications to Try

1. **Add Seasonality**
   - Modify demand based on season column
   - Use contextual bandits with season as context

2. **Multi-Supplier Coordination**
   - Optimize across all suppliers simultaneously
   - Balance lead time reliability vs cost

3. **Dynamic Pricing**
   - Let RL choose both discount and quantity
   - Maximize profit, not just fulfillment

4. **Risk Management**
   - Add CVaR (Conditional Value at Risk) constraints
   - Penalize high-variance strategies

## 📚 Connection to Theory

| RL Concept | Application Here |
|------------|------------------|
| Multi-Armed Bandit | Promotion discount selection |
| Epsilon-Greedy | Explore new truck combos vs exploit known good ones |
| UCB | Could replace epsilon-greedy for faster convergence |
| Contextual Bandit | Lead time varies by (supplier, SKU) context |
| Safety Stock | Like exploration - costly but prevents worse outcomes |

## 🏆 Success Metrics

Your solution is working well if:
- ✅ Promotion predictions within 10% of actual demand
- ✅ 90% of deliveries within predicted interval
- ✅ Truck utilization > 75% by both weight and volume
- ✅ Service level maintained > 95%
- ✅ Total cost (inventory + transport) minimized

## Next Steps

1. **Run the baseline**: `python grainger_combined_problem.py`
2. **Understand each component**: Read through the code
3. **Modify parameters**: Try different epsilon values, safety stock levels
4. **Add features**: Include weather, seasonality, competitor actions
5. **Try different algorithms**: UCB, Thompson Sampling, contextual bandits
6. **Build your own**: Create variations for your portfolio

---

*This problem combines real-world complexity with tractable RL solutions - perfect for demonstrating your skills in supply chain optimization roles!*

