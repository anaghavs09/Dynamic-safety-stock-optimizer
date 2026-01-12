"""
Compare All Three Methods
1. Manual (Traditional Formula)
2. Rule-Based (Industry Standard)
3. LLM-Powered (Experimental)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_all_methods(data_dir='data'):
    """
    Load results from all three methods and compare
    """
    
    print("\nSafety Stock Optimization: Three-Method Comparison Analysis")
    print("-" * 60)
    
    # Load results
    print("Loading method results...")
    manual = pd.read_csv(f'{data_dir}/manual_mode_results.csv')
    rule_based = pd.read_csv(f'{data_dir}/rule_based_results.csv')
    
    # Merge all results
    comparison = manual[['sku_id', 'description', 'abc_class', 'demand_pattern', 
                         'unit_cost', 'safety_stock', 'safety_stock_value', 
                         'annual_holding_cost']].copy()
    
    # Add rule-based results
    comparison = comparison.merge(
        rule_based[['sku_id', 'rule_based_safety_stock', 'rule_based_value', 
                    'rule_based_holding_cost', 'rule_based_reasoning']],
        on='sku_id', how='left'
    )
    
    # Try to load LLM results if available
    try:
        llm = pd.read_csv(f'{data_dir}/genai_assisted_results.csv')
        comparison = comparison.merge(
            llm[['sku_id', 'llm_safety_stock', 'llm_value', 
                 'llm_holding_cost', 'llm_reasoning']],
            on='sku_id', how='left'
        )
        has_llm = True
    except FileNotFoundError:
        print("Warning: LLM-powered results (genai_assisted_results.csv) not found. Skipping GenAI comparison.")
        has_llm = False
    
    # Calculate adjustment percentages for visualization and comparison
    comparison['rule_pct'] = ((comparison['rule_based_safety_stock'] - comparison['safety_stock']) / comparison['safety_stock'] * 100).fillna(0)
    if has_llm:
        comparison['llm_pct'] = ((comparison['llm_safety_stock'] - comparison['safety_stock']) / comparison['safety_stock'] * 100).fillna(0)
    
    print(f"Dataset size: {len(comparison)} SKUs")
    
    # --- Summary Metrics ---
    
    print("\nAggregated Performance Summary")
    print("-" * 60)
    
    methods = {
        'Manual': {
            'inventory': comparison['safety_stock_value'].sum(),
            'cost': comparison['annual_holding_cost'].sum(),
            'time': 'Instant'
        },
        'Rule-Based': {
            'inventory': comparison['rule_based_value'].sum(),
            'cost': comparison['rule_based_holding_cost'].sum(),
            'time': 'Instant'
        }
    }
    
    if has_llm:
        methods['LLM'] = {
            'inventory': comparison['llm_value'].sum(),
            'cost': comparison['llm_holding_cost'].sum(),
            'time': '3-5 minutes'
        }
    
    print("\n💰 INVENTORY VALUE:")
    for method, stats in methods.items():
        savings = comparison['safety_stock_value'].sum() - stats['inventory']
        savings_pct = (savings / comparison['safety_stock_value'].sum()) * 100
        print(f"   {method:12s}: ${stats['inventory']:>12,.2f}  ({savings_pct:>+6.1f}%)")
    
    print("\n💸 ANNUAL HOLDING COST:")
    for method, stats in methods.items():
        savings = comparison['annual_holding_cost'].sum() - stats['cost']
        savings_pct = (savings / comparison['annual_holding_cost'].sum()) * 100
        print(f"   {method:12s}: ${stats['cost']:>12,.2f}/year  ({savings_pct:>+6.1f}%)")
    
    print("\n⚡ PROCESSING TIME:")
    for method, stats in methods.items():
        print(f"   {method:12s}: {stats['time']}")
    
    # Note: Visualizations are now handled by visualizer_dashboard.py for higher quality output.
    
    # --- Detailed Analysis ---
    
    print("\nKey Optimization Insights")
    print("-" * 60)
    
    if has_llm:
        rule_savings = (comparison['safety_stock_value'].sum() - comparison['rule_based_value'].sum())
        llm_savings = (comparison['safety_stock_value'].sum() - comparison['llm_value'].sum())
        
        winner = "LLM" if llm_savings > rule_savings else "Rule-Based"
        win_amt = max(llm_savings, rule_savings)
        
        print(f"\n🏆 WINNER:")
        print(f"   {winner} Method saved ${win_amt:,.2f} ({win_amt/comparison['safety_stock_value'].sum()*100:.1f}%)")
        print(f"   (Rule-Based: ${rule_savings:,.2f} | LLM: ${llm_savings:,.2f})")
        
        # Agreement between methods
        agreement_threshold = 5  # Within 5% is "agreement"
        agreements = sum(abs(comparison['rule_pct'] - comparison['llm_pct']) < agreement_threshold)
        agreement_rate = agreements / len(comparison) * 100
        
        print(f"\n🤝 AGREEMENT:")
        print(f"   Rule-Based and LLM agree on {agreements}/{len(comparison)} SKUs ({agreement_rate:.1f}%)")
    
    # Most impactful optimizations
    comparison['rule_savings'] = comparison['safety_stock_value'] - comparison['rule_based_value']
    top_optimizations = comparison.nlargest(5, 'rule_savings')
    
    print(f"\nTop Optimization Opportunities (Rule-Based):")
    for _, row in top_optimizations.iterrows():
        print(f"  {row['sku_id']}: {row['description'][:40]}")
        print(f"    Baseline: {row['safety_stock']:,} | Adjusted: {row['rule_based_safety_stock']:,} units")
        print(f"    Estimated Savings: ${row['rule_savings']:,.2f}")
        print(f"    Rationale: {row['rule_based_reasoning'][:70]}...")
    
    # Save comprehensive comparison
    comparison_file = f'{data_dir}/complete_comparison.csv'
    comparison.to_csv(comparison_file, index=False)
    print(f"\nReport generated: {comparison_file}")
    print("-" * 60)
    print("Analysis complete.")
    
    return comparison


if __name__ == "__main__":
    results = compare_all_methods()