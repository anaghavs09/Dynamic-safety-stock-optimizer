"""
Data Generation Utilities for Dynamic Safety Stock Optimizer
Generates realistic retail supply chain data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


class RetailDataGenerator:
    """Generate realistic retail supply chain data"""
    
    def __init__(self, n_skus=100, n_weeks=52):
        self.n_skus = n_skus
        self.n_weeks = n_weeks
        self.categories = self._define_categories()
        
    def _define_categories(self):
        """Define product categories with characteristics"""
        return {
            'Electronics': {'price_range': (50, 500), 'lead_time': (4, 8), 'volatility': 0.3},
            'Clothing': {'price_range': (20, 150), 'lead_time': (3, 6), 'volatility': 0.4},
            'Home & Kitchen': {'price_range': (15, 200), 'lead_time': (2, 5), 'volatility': 0.25},
            'Toys': {'price_range': (10, 100), 'lead_time': (3, 7), 'volatility': 0.5},
            'Sports': {'price_range': (25, 300), 'lead_time': (3, 6), 'volatility': 0.35},
            'Beauty': {'price_range': (10, 80), 'lead_time': (2, 4), 'volatility': 0.3},
            'Grocery': {'price_range': (5, 50), 'lead_time': (1, 3), 'volatility': 0.2},
        }
    
    def generate_sku_master(self):
        """Generate SKU master data with realistic attributes"""
        
        skus = []
        sku_id = 1000
        
        # Distribution of demand patterns
        pattern_distribution = {
            'seasonal': 30,    # 30 seasonal items
            'trending': 20,    # 20 trending items
            'stable': 40,      # 40 stable items
            'erratic': 10      # 10 erratic items
        }
        
        for pattern, count in pattern_distribution.items():
            for i in range(count):
                # Pick a random category
                category = random.choice(list(self.categories.keys()))
                cat_info = self.categories[category]
                
                # Generate SKU attributes
                sku = {
                    'sku_id': f'SKU-{sku_id}',
                    'description': self._generate_product_name(category, pattern),
                    'category': category,
                    'demand_pattern': pattern,
                    'unit_cost': round(np.random.uniform(*cat_info['price_range']), 2),
                    'avg_weekly_demand': self._generate_avg_demand(pattern),
                    'demand_volatility': cat_info['volatility'],
                    'avg_lead_time_weeks': np.random.uniform(*cat_info['lead_time']),
                    'lead_time_std': np.random.uniform(0.5, 1.5),
                    'supplier_id': f'SUP-{random.randint(1, 20):03d}',
                    'unit_volume_cuft': round(np.random.uniform(0.5, 5), 2),
                }
                
                skus.append(sku)
                sku_id += 1
        
        df = pd.DataFrame(skus)
        
        # Add ABC classification based on annual value
        df['annual_value'] = df['avg_weekly_demand'] * 52 * df['unit_cost']
        df = self._classify_abc(df)
        
        return df
    
    def _generate_product_name(self, category, pattern):
        """Generate realistic product names"""
        prefixes = {
            'seasonal': ['Holiday', 'Seasonal', 'Winter', 'Summer'],
            'trending': ['New', 'Popular', 'Trending', 'Hot'],
            'stable': ['Classic', 'Essential', 'Standard', 'Basic'],
            'erratic': ['Limited', 'Promotional', 'Special', 'Flash']
        }
        
        products = {
            'Electronics': ['Headphones', 'Speaker', 'Charger', 'Cable', 'Mouse'],
            'Clothing': ['Jacket', 'Shirt', 'Pants', 'Dress', 'Sweater'],
            'Home & Kitchen': ['Cookware', 'Utensils', 'Towels', 'Bedding', 'Organizer'],
            'Toys': ['Action Figure', 'Puzzle', 'Board Game', 'Doll', 'Building Set'],
            'Sports': ['Yoga Mat', 'Dumbbells', 'Resistance Band', 'Water Bottle', 'Bag'],
            'Beauty': ['Moisturizer', 'Serum', 'Makeup', 'Cleanser', 'Mask'],
            'Grocery': ['Snacks', 'Beverage', 'Pasta', 'Sauce', 'Cereal']
        }
        
        prefix = random.choice(prefixes[pattern])
        product = random.choice(products[category])
        
        return f"{prefix} {product} - {category}"
    
    def _generate_avg_demand(self, pattern):
        """Generate average weekly demand based on pattern"""
        base_demands = {
            'seasonal': (50, 200),
            'trending': (100, 300),
            'stable': (80, 250),
            'erratic': (30, 150)
        }
        
        return round(np.random.uniform(*base_demands[pattern]), 0)
    
    def _classify_abc(self, df):
        """Classify SKUs using ABC analysis (Pareto principle)"""
        df = df.sort_values('annual_value', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage of total value
        cumulative_value = df['annual_value'].cumsum()
        total_value = df['annual_value'].sum()
        cumulative_pct = cumulative_value / total_value
        
        # A items: Top items representing ~70% of value
        # B items: Next items representing ~20% of value
        # C items: Remaining items representing ~10% of value
        df['abc_class'] = 'C'
        df.loc[cumulative_pct <= 0.90, 'abc_class'] = 'B'
        df.loc[cumulative_pct <= 0.70, 'abc_class'] = 'A'
        
        return df
    
    def generate_demand_history(self, sku_master):
        """Generate 52 weeks of demand history with realistic patterns"""
        
        demand_data = []
        
        for _, sku in sku_master.iterrows():
            # Generate demand series based on pattern type
            weekly_demand = self._generate_demand_series(
                pattern=sku['demand_pattern'],
                avg_demand=sku['avg_weekly_demand'],
                volatility=sku['demand_volatility'],
                n_weeks=self.n_weeks
            )
            
            # Store each week's demand
            for week, demand in enumerate(weekly_demand, 1):
                demand_data.append({
                    'sku_id': sku['sku_id'],
                    'week': week,
                    'demand': max(0, int(demand))  # No negative demand
                })
        
        return pd.DataFrame(demand_data)
    
    def _generate_demand_series(self, pattern, avg_demand, volatility, n_weeks):
        """Generate demand time series based on pattern type"""
        
        if pattern == 'seasonal':
            # Seasonal pattern with holiday peaks
            seasonal_component = self._seasonal_pattern(n_weeks, peak_weeks=[48, 49, 50, 51, 52])
            trend = np.ones(n_weeks)
            noise = np.random.normal(0, volatility * avg_demand, n_weeks)
            demand = avg_demand * seasonal_component * trend + noise
            
        elif pattern == 'trending':
            # Linear trend (growing or declining)
            trend_direction = random.choice([1, -1])
            trend = np.linspace(1, 1 + trend_direction * 0.5, n_weeks)
            noise = np.random.normal(0, volatility * avg_demand, n_weeks)
            demand = avg_demand * trend + noise
            
        elif pattern == 'stable':
            # Stable with low variation
            noise = np.random.normal(0, volatility * avg_demand, n_weeks)
            demand = avg_demand + noise
            
        else:  # erratic
            # High variability with random spikes
            base_demand = avg_demand * 0.7
            noise = np.random.normal(0, volatility * avg_demand, n_weeks)
            # Add random spikes
            spikes = np.random.choice([0, 1, 2, 3], n_weeks, p=[0.7, 0.2, 0.07, 0.03])
            demand = base_demand + noise + spikes * avg_demand
        
        return demand
    
    def _seasonal_pattern(self, n_weeks, peak_weeks):
        """Generate seasonal pattern with specific peak weeks"""
        pattern = np.ones(n_weeks) * 0.7  # Base level at 70%
        
        # Create peaks around holiday weeks
        for week in peak_weeks:
            if week <= n_weeks:
                pattern[week-1] = 2.0  # Double demand during peak
                # Gradual increase before peak
                if week > 3:
                    pattern[week-4:week-1] = np.linspace(0.7, 1.8, 3)
        
        return pattern
    
    def generate_lead_time_history(self, sku_master):
        """Generate lead time history for each SKU"""
        
        lead_time_data = []
        start_date = datetime(2024, 1, 1)
        
        for _, sku in sku_master.iterrows():
            # Generate 20-30 historical orders per SKU
            n_orders = random.randint(20, 30)
            
            for i in range(n_orders):
                order_date = start_date + timedelta(days=random.randint(0, 365))
                
                # Lead time with variability
                actual_lead_time = np.random.normal(
                    sku['avg_lead_time_weeks'] * 7,  # Convert weeks to days
                    sku['lead_time_std'] * 7
                )
                actual_lead_time = max(7, int(actual_lead_time))  # Minimum 1 week
                
                delivery_date = order_date + timedelta(days=actual_lead_time)
                
                lead_time_data.append({
                    'sku_id': sku['sku_id'],
                    'order_date': order_date.strftime('%Y-%m-%d'),
                    'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'lead_time_days': actual_lead_time
                })
        
        return pd.DataFrame(lead_time_data)
    
    def generate_cost_data(self, sku_master):
        """Generate cost-related data for each SKU"""
        
        cost_data = []
        
        for _, sku in sku_master.iterrows():
            # Holding cost rate varies by ABC class
            base_holding_rate = 0.25  # 25% annual holding cost
            
            # A items: Lower rate (better managed)
            # C items: Higher rate (less attention)
            if sku['abc_class'] == 'A':
                holding_rate = base_holding_rate * 0.8
            elif sku['abc_class'] == 'B':
                holding_rate = base_holding_rate
            else:
                holding_rate = base_holding_rate * 1.2
            
            # Stockout cost (opportunity cost of lost sale)
            if sku['demand_pattern'] in ['trending', 'seasonal']:
                stockout_multiplier = 1.5
            else:
                stockout_multiplier = 1.0
            
            stockout_cost = sku['unit_cost'] * 0.3 * stockout_multiplier
            
            cost_data.append({
                'sku_id': sku['sku_id'],
                'holding_cost_rate': round(holding_rate, 3),
                'stockout_cost': round(stockout_cost, 2),
                'order_cost': 50.0  # Fixed ordering cost
            })
        
        return pd.DataFrame(cost_data)


# Main function to generate all datasets
def generate_all_datasets(n_skus=100, n_weeks=52, output_dir='data'):
    """Generate all datasets and save to CSV files"""
    
    print(f"🔄 Generating retail supply chain data...")
    print(f"   - {n_skus} SKUs")
    print(f"   - {n_weeks} weeks of history\n")
    
    generator = RetailDataGenerator(n_skus=n_skus, n_weeks=n_weeks)
    
    # Generate datasets
    print("📦 Generating SKU master data...")
    sku_master = generator.generate_sku_master()
    
    print("📈 Generating demand history...")
    demand_history = generator.generate_demand_history(sku_master)
    
    print("🚚 Generating lead time history...")
    lead_time_history = generator.generate_lead_time_history(sku_master)
    
    print("💰 Generating cost data...")
    cost_data = generator.generate_cost_data(sku_master)
    
    # Save to CSV
    print(f"\n💾 Saving datasets to {output_dir}/")
    sku_master.to_csv(f'{output_dir}/sku_master.csv', index=False)
    demand_history.to_csv(f'{output_dir}/demand_history.csv', index=False)
    lead_time_history.to_csv(f'{output_dir}/lead_time_history.csv', index=False)
    cost_data.to_csv(f'{output_dir}/cost_data.csv', index=False)
    
    # Summary statistics
    print("\n✅ Data generation complete!")
    print("\n📊 Summary Statistics:")
    print(f"   Total SKUs: {len(sku_master)}")
    print(f"   Demand records: {len(demand_history):,}")
    print(f"   Lead time records: {len(lead_time_history):,}")
    print("\n   Demand Pattern Distribution:")
    print(sku_master['demand_pattern'].value_counts())
    print("\n   ABC Classification:")
    print(sku_master['abc_class'].value_counts())
    
    return sku_master, demand_history, lead_time_history, cost_data


if __name__ == "__main__":
    generate_all_datasets()