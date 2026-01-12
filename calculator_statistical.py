"""
Safety Stock Calculator - Manual Mode
Traditional formulas used by most companies
"""

import numpy as np
import pandas as pd
from scipy import stats


class ManualSafetyStockCalculator:
    """
    Calculate safety stock using traditional statistical formulas
    """
    
    def __init__(self, service_level=0.95):
        """
        Initialize calculator
        
        Parameters:
        -----------
        service_level : float
            Target service level (e.g., 0.95 = 95% in-stock probability)
        """
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
        
    def calculate_demand_statistics(self, demand_history):
        """
        Calculate demand statistics for each SKU
        
        Parameters:
        -----------
        demand_history : DataFrame
            Historical demand data with columns: sku_id, week, demand
            
        Returns:
        --------
        DataFrame with demand statistics per SKU
        """
        
        demand_stats = demand_history.groupby('sku_id').agg({
            'demand': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        demand_stats.columns = ['sku_id', 'avg_demand', 'std_demand', 'n_weeks', 'min_demand', 'max_demand']
        
        # Calculate coefficient of variation (CV)
        demand_stats['cv_demand'] = (demand_stats['std_demand'] / demand_stats['avg_demand']) * 100
        
        return demand_stats
    
    def calculate_lead_time_statistics(self, lead_time_history):
        """
        Calculate lead time statistics for each SKU
        
        Parameters:
        -----------
        lead_time_history : DataFrame
            Historical lead time data with columns: sku_id, lead_time_days
            
        Returns:
        --------
        DataFrame with lead time statistics per SKU
        """
        
        lt_stats = lead_time_history.groupby('sku_id').agg({
            'lead_time_days': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        lt_stats.columns = ['sku_id', 'avg_lead_time_days', 'std_lead_time_days', 
                            'n_orders', 'min_lead_time', 'max_lead_time']
        
        # Convert to weeks
        lt_stats['avg_lead_time_weeks'] = lt_stats['avg_lead_time_days'] / 7
        lt_stats['std_lead_time_weeks'] = lt_stats['std_lead_time_days'] / 7
        
        # Calculate coefficient of variation
        lt_stats['cv_lead_time'] = (lt_stats['std_lead_time_days'] / lt_stats['avg_lead_time_days']) * 100
        
        return lt_stats
    
    def calculate_safety_stock(self, sku_master, demand_stats, lead_time_stats):
        """
        Calculate safety stock using traditional formula
        
        Formula (assumes demand variability only):
        Safety Stock = Z × σ_demand × √(Lead Time)
        
        Where:
        - Z = Z-score for service level (e.g., 1.65 for 95%)
        - σ_demand = Standard deviation of demand
        - Lead Time = Average lead time in weeks
        
        Parameters:
        -----------
        sku_master : DataFrame
            SKU master data
        demand_stats : DataFrame
            Demand statistics per SKU
        lead_time_stats : DataFrame
            Lead time statistics per SKU
            
        Returns:
        --------
        DataFrame with safety stock calculations
        """
        
        # Merge all data
        results = sku_master[['sku_id', 'description', 'category', 'demand_pattern', 
                              'abc_class', 'unit_cost', 'unit_volume_cuft']].copy()
        
        results = results.merge(demand_stats[['sku_id', 'avg_demand', 'std_demand', 'cv_demand']], 
                                on='sku_id', how='left')
        results = results.merge(lead_time_stats[['sku_id', 'avg_lead_time_weeks', 'std_lead_time_weeks']], 
                                on='sku_id', how='left')
        
        # Calculate safety stock
        # Formula: SS = Z × σ_demand × √(LT)
        results['safety_stock'] = (
            self.z_score * 
            results['std_demand'] * 
            np.sqrt(results['avg_lead_time_weeks'])
        )
        
        # Round to whole units
        results['safety_stock'] = results['safety_stock'].round(0).astype(int)
        
        # Calculate reorder point (ROP)
        # Formula: ROP = (Avg Demand × Lead Time) + Safety Stock
        results['reorder_point'] = (
            (results['avg_demand'] * results['avg_lead_time_weeks']) + 
            results['safety_stock']
        )
        results['reorder_point'] = results['reorder_point'].round(0).astype(int)
        
        # Calculate average demand during lead time
        results['demand_during_lead_time'] = (
            results['avg_demand'] * results['avg_lead_time_weeks']
        ).round(0).astype(int)
        
        # Store parameters used
        results['service_level'] = self.service_level
        results['z_score'] = self.z_score
        
        return results
    
    def calculate_inventory_costs(self, results, cost_data):
        """
        Calculate inventory holding costs
        
        Parameters:
        -----------
        results : DataFrame
            Safety stock calculation results
        cost_data : DataFrame
            Cost data per SKU
            
        Returns:
        --------
        DataFrame with cost calculations
        """
        
        # Merge cost data
        results = results.merge(cost_data[['sku_id', 'holding_cost_rate', 'stockout_cost']], 
                               on='sku_id', how='left')
        
        # Annual holding cost for safety stock
        # Formula: Safety Stock × Unit Cost × Holding Cost Rate
        results['annual_holding_cost'] = (
            results['safety_stock'] * 
            results['unit_cost'] * 
            results['holding_cost_rate']
        )
        
        # Total inventory value (just safety stock)
        results['safety_stock_value'] = (
            results['safety_stock'] * results['unit_cost']
        )
        
        # Total warehouse space required
        results['warehouse_space_cuft'] = (
            results['safety_stock'] * results['unit_volume_cuft']
        )
        
        return results
    
    def generate_summary_statistics(self, results):
        """
        Generate summary statistics across all SKUs
        
        Parameters:
        -----------
        results : DataFrame
            Complete calculation results
            
        Returns:
        --------
        Dictionary with summary metrics
        """
        
        summary = {
            'total_skus': len(results),
            'total_safety_stock_units': results['safety_stock'].sum(),
            'total_safety_stock_value': results['safety_stock_value'].sum(),
            'total_annual_holding_cost': results['annual_holding_cost'].sum(),
            'total_warehouse_space': results['warehouse_space_cuft'].sum(),
            'avg_safety_stock_per_sku': results['safety_stock'].mean(),
            'median_safety_stock': results['safety_stock'].median(),
            'service_level': self.service_level,
            'z_score': self.z_score
        }
        
        # Summary by ABC class
        abc_summary = results.groupby('abc_class').agg({
            'safety_stock': 'sum',
            'safety_stock_value': 'sum',
            'annual_holding_cost': 'sum',
            'sku_id': 'count'
        }).reset_index()
        abc_summary.columns = ['abc_class', 'total_safety_stock', 'total_value', 
                               'total_holding_cost', 'number_of_skus']
        
        # Summary by demand pattern
        pattern_summary = results.groupby('demand_pattern').agg({
            'safety_stock': 'sum',
            'safety_stock_value': 'sum',
            'annual_holding_cost': 'sum',
            'sku_id': 'count'
        }).reset_index()
        pattern_summary.columns = ['demand_pattern', 'total_safety_stock', 'total_value', 
                                   'total_holding_cost', 'number_of_skus']
        
        summary['abc_breakdown'] = abc_summary
        summary['pattern_breakdown'] = pattern_summary
        
        return summary


def run_manual_mode_calculation(data_dir='data', service_level=0.95):
    """
    Complete manual mode calculation workflow
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    service_level : float
        Target service level
        
    Returns:
    --------
    results : DataFrame
        Complete calculation results
    summary : dict
        Summary statistics
    """
    
    print("="*80)
    print("👋 STEP 1: Running the traditional math baseline...")
    print("="*80)
    print(f"\nTarget Service Level: {service_level*100}%")
    print(f"Standard Multiplier (Z-Score): {stats.norm.ppf(service_level):.3f}\n")
    
    # Load data
    print("📦 Grabbing your data...")
    sku_master = pd.read_csv(f'{data_dir}/sku_master.csv')
    demand_history = pd.read_csv(f'{data_dir}/demand_history.csv')
    lead_time_history = pd.read_csv(f'{data_dir}/lead_time_history.csv')
    cost_data = pd.read_csv(f'{data_dir}/cost_data.csv')
    print(f"   ✓ Loaded {len(sku_master)} SKUs")
    print(f"   ✓ Loaded {len(demand_history):,} demand records")
    print(f"   ✓ Loaded {len(lead_time_history):,} lead time records\n")
    
    # Initialize calculator
    calculator = ManualSafetyStockCalculator(service_level=service_level)
    
    # Calculate statistics
    print("📊 Calculating demand statistics...")
    demand_stats = calculator.calculate_demand_statistics(demand_history)
    
    print("🚚 Calculating lead time statistics...")
    lead_time_stats = calculator.calculate_lead_time_statistics(lead_time_history)
    
    # Calculate safety stock
    print("🔢 Calculating safety stock using formula:")
    print("   Safety Stock = Z × σ_demand × √(Lead Time)")
    results = calculator.calculate_safety_stock(sku_master, demand_stats, lead_time_stats)
    
    # Calculate costs
    print("💰 Calculating inventory costs...\n")
    results = calculator.calculate_inventory_costs(results, cost_data)
    
    # Generate summary
    summary = calculator.generate_summary_statistics(results)
    
    # Print summary
    print("="*80)
    print("CALCULATION RESULTS")
    print("="*80)
    print(f"\n📦 Total SKUs: {summary['total_skus']}")
    print(f"📊 Total Safety Stock: {summary['total_safety_stock_units']:,} units")
    print(f"💵 Total Safety Stock Value: ${summary['total_safety_stock_value']:,.2f}")
    print(f"💸 Annual Holding Cost: ${summary['total_annual_holding_cost']:,.2f}")
    print(f"📏 Warehouse Space Required: {summary['total_warehouse_space']:,.0f} cubic feet")
    
    print("\n" + "="*80)
    print("BREAKDOWN BY ABC CLASS")
    print("="*80)
    print(summary['abc_breakdown'].to_string(index=False))
    
    print("\n" + "="*80)
    print("BREAKDOWN BY DEMAND PATTERN")
    print("="*80)
    print(summary['pattern_breakdown'].to_string(index=False))
    
    # Save results
    output_file = f'{data_dir}/manual_mode_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✨ STEP 1 COMPLETE: The math baseline is ready!")
    print("="*80)
    
    return results, summary


if __name__ == "__main__":
    results, summary = run_manual_mode_calculation()