"""
GenAI-Assisted Safety Stock Calculator
Uses Google Gemini API with batch processing for efficiency
Part of Three-Method Comparison Project
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import re
import google.generativeai as genai
import time
import json
import random
import asyncio

# Load environment variables
load_dotenv()


class DynamicBatchManager:
    """Manages batch size dynamically based on API feedback"""
    def __init__(self, initial_size=50, min_size=10, max_size=100):
        self.batch_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.success_streak = 0
        
    def adjust_for_error(self):
        """Decrease batch size on 429 errors to mitigate rate limit impact."""
        self.batch_size = max(self.min_size, int(self.batch_size * 0.5))
        self.success_streak = 0
        print(f"Batch size reduced to {self.batch_size}")
        
    def adjust_for_success(self):
        """Incrementally increase batch size after successful requests."""
        self.success_streak += 1
        if self.success_streak >= 2 and self.batch_size < self.max_size:
            self.batch_size = min(self.max_size, self.batch_size + 10)
            self.success_streak = 0
            print(f"Batch size increased to {self.batch_size}")

async def call_Gemini_async(prompt, system_prompt="You are a supply chain optimization expert."):
    """
    Call Google Gemini API Asynchronously
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Gemini API key not found.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        generation_config={
            'temperature': 0.1,
            'max_output_tokens': 2000,
            'response_mime_type': 'application/json'
        }
    )
    
    full_prompt = f"{system_prompt}\n\n{prompt}"
    response = await model.generate_content_async(full_prompt)
    return response.text.strip()


def calculate_demand_statistics(demand_history):
    """Calculate comprehensive demand statistics"""
    
    avg_demand = demand_history['demand'].mean()
    std_demand = demand_history['demand'].std()
    cv = (std_demand / avg_demand) * 100 if avg_demand > 0 else 0
    
    # Calculate trend
    x = np.arange(len(demand_history))
    y = demand_history['demand'].values
    slope = np.polyfit(x, y, 1)[0]
    
    avg = y.mean()
    if avg == 0:
        trend_direction = "stable"
    else:
        pct_change = (slope * len(x) / avg) * 100
        if abs(pct_change) < 5:
            trend_direction = "stable"
        elif pct_change > 5:
            trend_direction = f"increasing (+{pct_change:.1f}%)"
        else:
            trend_direction = f"decreasing ({pct_change:.1f}%)"
    
    # Recent vs overall
    recent_4wk = demand_history.tail(4)['demand'].mean()
    recent_vs_avg = recent_4wk / avg_demand if avg_demand > 0 else 1
    
    return {
        'avg_demand': avg_demand,
        'std_demand': std_demand,
        'cv': cv,
        'trend': trend_direction,
        'recent_vs_avg': recent_vs_avg,
        'recent_weeks': demand_history.tail(12)['demand'].tolist()
    }


async def batch_genai_optimization_async(batch_data, batch_manager):
    """
    Process multiple SKUs asynchronously with dynamic batching.
    """
    items_info = []
    for sku_id, sku_info in batch_data.items():
        items_info.append({
            "sku_id": sku_id,
            "name": sku_info['description'],
            "pattern": sku_info['demand_pattern'],
            "avg_demand": round(sku_info['demand_stats']['avg_demand'], 1),
            "cv": round(sku_info['demand_stats']['cv'], 1),
            "trend": sku_info['demand_stats']['trend'],
            "manual_ss": int(sku_info['manual_ss'])
        })

    prompt = f"""
    Analyze {len(items_info)} products and suggest safety stock adjustments (-50% to +50%).
    JSON Data: {json.dumps(items_info)}
    Return JSON: {{sku_id: {{"adjustment_pct": int, "reasoning": "string"}}, ...}}
    """
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            response_text = await call_Gemini_async(
                prompt, 
                "You are an inventory optimization API. Return ONLY raw JSON."
            )
            
            try:
                results = json.loads(response_text)
                adjustments = {}
                justifications = {}
                
                for sku_id, data in results.items():
                    if sku_id in batch_data:
                        adj = data.get('adjustment_pct', 0)
                        adj = max(-50, min(50, int(adj)))
                        adjustments[sku_id] = adj
                        justifications[sku_id] = data.get('reasoning', "AI recommendation")[:100]
                
                batch_manager.adjust_for_success()
                return adjustments, justifications
                
            except json.JSONDecodeError:
                if attempt == max_retries:
                    return {sku_id: None for sku_id in batch_data.keys()}, {sku_id: "JSON Error" for sku_id in batch_data.keys()}
                await asyncio.sleep(1)
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                batch_manager.adjust_for_error()
                if attempt < max_retries:
                     wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    return {sku_id: None for sku_id in batch_data.keys()}, {sku_id: "Rate Limit Exceeded" for sku_id in batch_data.keys()}
            else:
                print(f"API Error: {error_msg[:80]}")
                return {sku_id: None for sku_id in batch_data.keys()}, {sku_id: "API Error" for sku_id in batch_data.keys()}

    return {sku_id: None for sku_id in batch_data.keys()}, {sku_id: "Unknown Error" for sku_id in batch_data.keys()}


def fallback_rule_based_adjustment(sku_data, demand_stats, cv):
    """
    Fallback rule-based logic when API fails
    """
    adjustment_pct = 0
    reasoning = "Standard formula applied."
    
    # High variability → increase
    if cv > 50:
        adjustment_pct = 15
        reasoning = "High variability. Increasing buffer."
    
    # Trending up → increase
    elif "increasing" in demand_stats['trend'].lower():
        adjustment_pct = 20
        reasoning = "Upward trend. Increasing for growth."
    
    # Trending down → decrease
    elif "decreasing" in demand_stats['trend'].lower():
        adjustment_pct = -15
        reasoning = "Downward trend. Reducing excess."
    
    # Seasonal peak approaching
    elif sku_data['demand_pattern'] == 'seasonal':
        if demand_stats['recent_vs_avg'] > 1.3:
            adjustment_pct = 25
            reasoning = "Seasonal peak approaching."
    
    # Stable items → can reduce
    elif sku_data['demand_pattern'] == 'stable' and cv < 20:
        adjustment_pct = -10
        reasoning = "Very stable. Can reduce safely."
    
    return adjustment_pct, reasoning


async def run_genai_assisted_mode(data_dir='data', manual_results_file=None, batch_size=50):
    """
    Run GenAI-assisted safety stock calculation asynchronously
    """
    
    print("\nGenAI-Assisted Optimization Strategy (Gemini)")
    print("-" * 60)
    
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
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️  Gemini API key not found! Set GEMINI_API_KEY.")
        return None
    
    # Initialize results
    results = manual_results.copy()
    results['llm_adjustment_pct'] = 0
    results['llm_reasoning'] = ""
    results['llm_safety_stock'] = 0
    results['llm_method'] = ""
    
    # Dynamic Batching State
    batch_manager = DynamicBatchManager(initial_size=batch_size)
    semaphore = asyncio.Semaphore(2) # Max 2 concurrent API calls
    
    all_sku_ids = results['sku_id'].tolist()
    processed_count = 0
    
    print(f"\nProcessing {len(results)} SKUs using asynchronous batching...")
    
    while processed_count < len(results):
        current_batch_size = batch_manager.batch_size
        end_idx = min(processed_count + current_batch_size, len(results))
        batch_sku_ids = all_sku_ids[processed_count:end_idx]
        
        # Prepare batch data
        batch_data = {}
        for sku_id in batch_sku_ids:
            row = results[results['sku_id'] == sku_id].iloc[0]
            sku_info = sku_master[sku_master['sku_id'] == sku_id].iloc[0]
            sku_demand = demand_history[demand_history['sku_id'] == sku_id]
            sku_lt = lead_time_stats[lead_time_stats['sku_id'] == sku_id].iloc[0]
            
            demand_stats = calculate_demand_statistics(sku_demand)
            
            batch_data[sku_id] = {
                'description': sku_info['description'],
                'demand_pattern': sku_info['demand_pattern'],
                'demand_stats': demand_stats,
                'manual_ss': row['safety_stock']
            }
        
        async def process_with_semaphore():
            async with semaphore:
                return await batch_genai_optimization_async(batch_data, batch_manager)
        
        print(f"Processing batch ({len(batch_sku_ids)} SKUs) - Current batch size: {current_batch_size}")
        adjustments, justifications = await process_with_semaphore()
        
        # Store results for this batch
        for sku_id in batch_sku_ids:
            idx = results[results['sku_id'] == sku_id].index[0]
            if sku_id in adjustments and adjustments[sku_id] is not None:
                adjustment = adjustments[sku_id]
                justification = justifications[sku_id]
                method = "LLM"
            else:
                # Fallback for this SKU if LLM failed
                sku_info = sku_master[sku_master['sku_id'] == sku_id].iloc[0]
                demand_stats = batch_data[sku_id]['demand_stats']
                adjustment, justification = fallback_rule_based_adjustment(
                    sku_info, demand_stats, demand_stats['cv']
                )
                method = "Fallback"
            
            results.at[idx, 'llm_adjustment_pct'] = adjustment
            results.at[idx, 'llm_reasoning'] = justification
            results.at[idx, 'llm_method'] = method
            
            # Calculate adjusted safety stock
            adjustment_factor = 1 + (adjustment / 100)
            adjusted_ss = results.at[idx, 'safety_stock'] * adjustment_factor
            results.at[idx, 'llm_safety_stock'] = int(round(adjusted_ss))
            
        processed_count = end_idx
    
    print(f"\n   ✓ All SKUs processed!")
    
    # Recalculate costs and metrics
    results['llm_value'] = results['llm_safety_stock'] * results['unit_cost']
    results['llm_holding_cost'] = results['llm_value'] * results['holding_cost_rate']
    results['llm_warehouse_space'] = results['llm_safety_stock'] * results['unit_volume_cuft']
    results['llm_value_savings'] = results['safety_stock_value'] - results['llm_value']
    results['llm_cost_savings'] = results['annual_holding_cost'] - results['llm_holding_cost']
    results['llm_space_savings'] = results['warehouse_space_cuft'] - results['llm_warehouse_space']
    
    # Save results
    output_file = f'{data_dir}/genai_assisted_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\nAI-assisted optimization complete.")
    print("-" * 60)
    
    return results

if __name__ == "__main__":
    asyncio.run(run_genai_assisted_mode(batch_size=50))