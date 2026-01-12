"""
Rule-Based Safety Stock Optimizer
Industry-standard approach used by leading retailers
"""

import pandas as pd
import numpy as np


def calculate_demand_statistics(demand_history):
    """Calculate comprehensive demand statistics"""
    
    avg_demand = demand_history['demand'].mean()
    std_demand = demand_history['demand'].std()
    cv = (std_demand / avg_demand) * 100 if avg_demand > 0 else 0
    
    # Trend calculation
    x = np.arange(len(demand_history))
    y = demand_history['demand'].values
    slope = np.polyfit(x, y, 1)[0]
    trend_pct = (slope * len(x) / avg_demand) * 100 if avg_demand > 0 else 0
    
    # Recent vs overall comparison
    recent_4wk = demand_history.tail(4)['demand'].mean()
    recent_vs_avg = recent_4wk / avg_demand if avg_demand > 0 else 1
    
    return {
        'avg_demand': avg_demand,
        'std_demand': std_demand,
        'cv': cv,
        'trend_pct': trend_pct,
        'recent_vs_avg': recent_vs_avg
    }


def rule_based_optimization(sku_data, demand_stats, lead_time_stats, manual_safety_stock):
    """
    Multi-factor rule-based optimization
    Based on industry best practices
    
    Parameters:
    -----------
    sku_data : Series
        SKU master data
    demand_stats : dict
        Demand statistics
    lead_time_stats : Series
        Lead time statistics
    manual_safety_stock : int
        Traditional formula result
        
    Returns:
    --------
    optimized_ss : int
        Optimized safety stock
    adjustment_pct : int
        Percentage adjustment
    reasoning : str
        Explanation of decision
    """
    
    adjustment_factor = 1.0
    reasons = []
    
    # ═══════════════════════════════════════════════════════
    # RULE SET 1: ABC Classification Strategy
    # ═══════════════════════════════════════════════════════
    
    if sku_data['abc_class'] == 'A':
        # A-items: High-value, optimize carefully
        if demand_stats['cv'] < 15:
            # Very stable A-item → Can reduce safely
            adjustment_factor *= 0.85
            reasons.append(f"A-item + stable demand (CV={demand_stats['cv']:.1f}%): -15%")
        
        elif demand_stats['cv'] > 40:
            # Highly variable A-item → Need more buffer
            adjustment_factor *= 1.20
            reasons.append(f"A-item + high variability (CV={demand_stats['cv']:.1f}%): +20%")
        
        else:
            # Moderate variability → Keep standard
            reasons.append(f"A-item + moderate variability: maintain")
    
    elif sku_data['abc_class'] == 'C':
        # C-items: Low-value, be aggressive with reduction
        if demand_stats['cv'] < 25:
            # Stable C-item → Reduce significantly
            adjustment_factor *= 0.75
            reasons.append(f"C-item + stable demand (CV={demand_stats['cv']:.1f}%): -25%")
        
        elif demand_stats['cv'] > 50:
            # Erratic C-item → Small increase (not worth stockouts)
            adjustment_factor *= 1.10
            reasons.append(f"C-item + high variability (CV={demand_stats['cv']:.1f}%): +10%")
        
        else:
            # Moderate → Reduce moderately
            adjustment_factor *= 0.85
            reasons.append(f"C-item: -15%")
    
    else:  # B-items
        # B-items: Balanced approach
        if demand_stats['cv'] > 45:
            adjustment_factor *= 1.15
            reasons.append(f"B-item + high variability: +15%")
        
        elif demand_stats['cv'] < 20:
            adjustment_factor *= 0.88
            reasons.append(f"B-item + stable demand: -12%")
        
        else:
            reasons.append(f"B-item: maintain")
    
    # ═══════════════════════════════════════════════════════
    # RULE SET 2: Demand Pattern Analysis
    # ═══════════════════════════════════════════════════════
    
    if sku_data['demand_pattern'] == 'seasonal':
        # Seasonal items: Check if in peak or off-season
        if demand_stats['recent_vs_avg'] > 1.5:
            # Peak season detected
            adjustment_factor *= 1.35
            reasons.append(f"Seasonal PEAK (recent {demand_stats['recent_vs_avg']:.1f}x avg): +35%")
        
        elif demand_stats['recent_vs_avg'] < 0.7:
            # Off-season detected
            adjustment_factor *= 0.70
            reasons.append(f"Seasonal OFF-SEASON (recent {demand_stats['recent_vs_avg']:.1f}x avg): -30%")
        
        else:
            # Normal seasonal period
            reasons.append(f"Seasonal item in normal period")
    
    elif sku_data['demand_pattern'] == 'trending':
        # Trending items: Adjust for growth/decline
        if demand_stats['trend_pct'] > 15:
            # Strong growth
            adjustment_factor *= 1.25
            reasons.append(f"Strong GROWTH trend (+{demand_stats['trend_pct']:.1f}%): +25%")
        
        elif demand_stats['trend_pct'] < -15:
            # Strong decline
            adjustment_factor *= 0.75
            reasons.append(f"Strong DECLINE trend ({demand_stats['trend_pct']:.1f}%): -25%")
        
        elif abs(demand_stats['trend_pct']) > 8:
            # Moderate trend
            factor = 1.15 if demand_stats['trend_pct'] > 0 else 0.85
            adjustment_factor *= factor
            direction = "growth" if demand_stats['trend_pct'] > 0 else "decline"
            reasons.append(f"Moderate {direction} trend: {'+15%' if factor > 1 else '-15%'}")
    
    elif sku_data['demand_pattern'] == 'stable':
        # Stable items: Can reduce safety stock
        adjustment_factor *= 0.88
        reasons.append(f"Stable demand pattern: -12%")
    
    elif sku_data['demand_pattern'] == 'erratic':
        # Erratic items: Need more buffer
        if demand_stats['cv'] > 60:
            adjustment_factor *= 1.25
            reasons.append(f"Highly erratic (CV={demand_stats['cv']:.1f}%): +25%")
        else:
            adjustment_factor *= 1.12
            reasons.append(f"Erratic pattern: +12%")
    
    # ═══════════════════════════════════════════════════════
    # RULE SET 3: Lead Time Reliability
    # ═══════════════════════════════════════════════════════
    
    lt_cv = (lead_time_stats['std_lead_time_weeks'] / lead_time_stats['avg_lead_time_weeks']) * 100
    
    if lt_cv > 35:
        # Very unreliable supplier
        adjustment_factor *= 1.15
        reasons.append(f"Unreliable supplier (LT CV={lt_cv:.1f}%): +15%")
    
    elif lt_cv < 12:
        # Very reliable supplier
        adjustment_factor *= 0.93
        reasons.append(f"Reliable supplier (LT CV={lt_cv:.1f}%): -7%")
    
    # ═══════════════════════════════════════════════════════
    # CALCULATE FINAL RESULT
    # ═══════════════════════════════════════════════════════
    
    # Limit adjustment factor to reasonable range (0.5x to 1.5x)
    adjustment_factor = max(0.5, min(1.5, adjustment_factor))
    
    optimized_ss = int(manual_safety_stock * adjustment_factor)
    adjustment_pct = int((adjustment_factor - 1) * 100)
    
    # Combine reasoning
    if len(reasons) == 0:
        reasoning = "Standard formula maintained"
    elif len(reasons) == 1:
        reasoning = reasons[0]
    else:
        reasoning = " → ".join(reasons[:3])  # Top 3 reasons
    
    return optimized_ss, adjustment_pct, reasoning


def run_rule_based_optimization(data_dir='data', manual_results_file=None):
    """
    Run rule-based safety stock optimization on all SKUs
    """
    
    print("\n" + "="*80)
    print("🧠 STEP 2: Applying industry 'gut feel' (Rule-Based)")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data files...")
    sku_master = pd.read_csv(f'{data_dir}/sku_master.csv')
    demand_history = pd.read_csv(f'{data_dir}/demand_history.csv')
    
    # Load lead time stats
    from calculator_statistical import ManualSafetyStockCalculator
    calculator = ManualSafetyStockCalculator()
    lead_time_history = pd.read_csv(f'{data_dir}/lead_time_history.csv')
    lead_time_stats = calculator.calculate_lead_time_statistics(lead_time_history)
    
    # Load manual mode results
    if manual_results_file is None:
        manual_results_file = f'{data_dir}/manual_mode_results.csv'
    
    manual_results = pd.read_csv(manual_results_file)
    print(f"   ✓ Loaded {len(manual_results)} SKUs")
    
    # Initialize results
    results = manual_results.copy()
    results['rule_based_safety_stock'] = 0
    results['rule_based_adjustment_pct'] = 0
    results['rule_based_reasoning'] = ""
    
    print(f"\n🔧 Applying rule-based optimization...")
    
    # Process each SKU (INSTANT - no API calls!)
    for idx, row in results.iterrows():
        sku_id = row['sku_id']
        
        # Get SKU-specific data
        sku_demand = demand_history[demand_history['sku_id'] == sku_id]
        sku_info = sku_master[sku_master['sku_id'] == sku_id].iloc[0]
        sku_lt = lead_time_stats[lead_time_stats['sku_id'] == sku_id].iloc[0]
        
        # Calculate demand statistics
        demand_stats = calculate_demand_statistics(sku_demand)
        
        # Apply rule-based optimization
        optimized_ss, adjustment_pct, reasoning = rule_based_optimization(
            sku_info, demand_stats, sku_lt, row['safety_stock']
        )
        
        # Store results
        results.at[idx, 'rule_based_safety_stock'] = optimized_ss
        results.at[idx, 'rule_based_adjustment_pct'] = adjustment_pct
        results.at[idx, 'rule_based_reasoning'] = reasoning
        
        # Show progress
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(results)} SKUs...")
    
    print(f"   ✓ All SKUs processed in <1 second!")
    
    # Recalculate costs
    results['rule_based_value'] = results['rule_based_safety_stock'] * results['unit_cost']
    results['rule_based_holding_cost'] = results['rule_based_value'] * results['holding_cost_rate']
    results['rule_based_warehouse_space'] = results['rule_based_safety_stock'] * results['unit_volume_cuft']
    
    # Calculate savings
    results['rule_based_value_savings'] = results['safety_stock_value'] - results['rule_based_value']
    results['rule_based_cost_savings'] = results['annual_holding_cost'] - results['rule_based_holding_cost']
    results['rule_based_space_savings'] = results['warehouse_space_cuft'] - results['rule_based_warehouse_space']
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS: MANUAL vs RULE-BASED")
    print("="*80)
    
    manual_total = results['safety_stock_value'].sum()
    rule_total = results['rule_based_value'].sum()
    savings = results['rule_based_value_savings'].sum()
    savings_pct = (savings / manual_total) * 100
    
    print(f"\n💰 INVENTORY VALUE:")
    print(f"   Manual Method:        ${manual_total:,.2f}")
    print(f"   Rule-Based Method:    ${rule_total:,.2f}")
    print(f"   Savings:              ${savings:,.2f} ({savings_pct:+.1f}%)")
    
    manual_cost = results['annual_holding_cost'].sum()
    rule_cost = results['rule_based_holding_cost'].sum()
    cost_savings = results['rule_based_cost_savings'].sum()
    cost_savings_pct = (cost_savings / manual_cost) * 100
    
    print(f"\n💸 ANNUAL HOLDING COST:")
    print(f"   Manual Method:        ${manual_cost:,.2f}/year")
    print(f"   Rule-Based Method:    ${rule_cost:,.2f}/year")
    print(f"   Savings:              ${cost_savings:,.2f}/year ({cost_savings_pct:+.1f}%)")
    
    # Adjustments breakdown
    increased = (results['rule_based_adjustment_pct'] > 0).sum()
    decreased = (results['rule_based_adjustment_pct'] < 0).sum()
    maintained = (results['rule_based_adjustment_pct'] == 0).sum()
    
    print(f"\n📈 ADJUSTMENTS:")
    print(f"   Increased safety stock:  {increased} SKUs")
    print(f"   Decreased safety stock:  {decreased} SKUs")
    print(f"   Maintained:              {maintained} SKUs")
    
    # Top savings
    print(f"\n🏆 TOP 5 OPTIMIZATION OPPORTUNITIES:")
    top_savers = results.nlargest(5, 'rule_based_value_savings')[
        ['sku_id', 'description', 'abc_class', 'demand_pattern',
         'safety_stock', 'rule_based_safety_stock', 'rule_based_value_savings', 'rule_based_reasoning']
    ]
    
    for _, row in top_savers.iterrows():
        print(f"\n   {row['sku_id']}: {row['description'][:40]}")
        print(f"   Class: {row['abc_class']} | Pattern: {row['demand_pattern']}")
        print(f"   Manual: {row['safety_stock']} → Rule-Based: {row['rule_based_safety_stock']} units")
        print(f"   Savings: ${row['rule_based_value_savings']:,.2f}")
        print(f"   Logic: {row['rule_based_reasoning'][:80]}")
    
    # Save results
    output_file = f'{data_dir}/rule_based_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ STEP 2 COMPLETE: The heuristic layer is all set!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_rule_based_optimization()