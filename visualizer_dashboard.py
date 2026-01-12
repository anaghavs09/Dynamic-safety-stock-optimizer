import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

def generate_visuals(data_dir='data', output_dir='data/visuals'):
    """
    Generate static and interactive visualizations for method comparison
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load data
    df = pd.read_csv(f'{data_dir}/complete_comparison.csv')
    
    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. STATIC DASHBOARD
    # ═══════════════════════════════════════════════════════
    
    print("Generating static dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Color Palette
    colors = {'Manual': '#6C757D', 'Rule-Based': '#FF7F50', 'LLM': '#20B2AA'}
    
    # A. Total Annual Holding Cost Comparison
    costs = {
        'Manual': df['annual_holding_cost'].sum(),
        'Rule-Based': df['rule_based_holding_cost'].sum(),
        'LLM': df['llm_holding_cost'].sum()
    }
    cost_df = pd.DataFrame(list(costs.items()), columns=['Method', 'Cost'])
    
    sns.barplot(x='Method', y='Cost', data=cost_df, palette=list(colors.values()), ax=axes[0,0])
    axes[0,0].set_title('Total Annual Holding Cost', fontsize=18, fontweight='bold', pad=20)
    axes[0,0].set_ylabel('Annual Cost ($)', fontsize=14)
    axes[0,0].set_xlabel('', fontsize=14)
    for i, v in enumerate(cost_df['Cost']):
        axes[0,0].text(i, v + 5000, f'${v:,.0f}', ha='center', fontsize=12, fontweight='bold')

    # B. Distribution of Adjustments (%)
    sns.kdeplot(df['rule_pct'], fill=True, color=colors['Rule-Based'], label='Rule-Based', ax=axes[0,1])
    sns.kdeplot(df['llm_pct'], fill=True, color=colors['LLM'], label='LLM', ax=axes[0,1])
    axes[0,1].set_title('Distribution of Optimization Adjustments', fontsize=18, fontweight='bold', pad=20)
    axes[0,1].set_xlabel('Adjustment from Manual (%)', fontsize=14)
    axes[0,1].legend(fontsize=12)
    axes[0,1].axvline(0, color='black', linestyle='--', alpha=0.3)

    # C. Savings by ABC Class
    abc_savings = df.groupby('abc_class').agg({
        'rule_savings': 'sum',
        'llm_value': lambda x: df.loc[x.index, 'safety_stock_value'].sum() - x.sum()
    }).reset_index()
    abc_savings.columns = ['ABC Class', 'Rule-Based Savings', 'LLM Savings']
    
    abc_melted = abc_savings.melt(id_vars='ABC Class', var_name='Method', value_name='Savings')
    sns.barplot(x='ABC Class', y='Savings', hue='Method', data=abc_melted, palette=[colors['Rule-Based'], colors['LLM']], ax=axes[1,0])
    axes[1,0].set_title('Inventory Value Savings by ABC Class', fontsize=18, fontweight='bold', pad=20)
    axes[1,0].set_ylabel('Total Savings ($)', fontsize=14)
    axes[1,0].legend(fontsize=12)

    # D. Method Precision (LLM vs Rule Agreement)
    sns.scatterplot(x='rule_pct', y='llm_pct', data=df, hue='abc_class', style='abc_class', 
                    palette='viridis', s=100, alpha=0.7, ax=axes[1,1])
    axes[1,1].set_title('Logic Alignment: LLM vs Rule-Based', fontsize=18, fontweight='bold', pad=20)
    axes[1,1].set_xlabel('Rule-Based Adjustment (%)', fontsize=14)
    axes[1,1].set_ylabel('LLM Adjustment (%)', fontsize=14)
    axes[1,1].plot([-30, 60], [-30, 60], color='red', linestyle='--', alpha=0.3) # 1:1 Line

    plt.tight_layout(pad=5.0)
    static_output = f'{output_dir}/dashboard.png'
    plt.savefig(static_output, dpi=300, bbox_inches='tight')
    print(f"Static dashboard saved to {static_output}")

    # 2. INTERACTIVE DASHBOARD
    # ═══════════════════════════════════════════════════════
    
    print("Generating interactive Plotly dashboard...")
    
    # Create the figure with multiple subplots
    fig_int = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Annual Cost Comparison", "SKU Adjustment Distribution", 
                        "Savings Waterfall (Manual → LLM)", "SKU Detail Matrix"),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "waterfall"}, {"type": "scatter"}]]
    )

    # 1. Bar Chart: Costs
    fig_int.add_trace(
        go.Bar(x=['Manual', 'Rule-Based', 'LLM'], y=[costs['Manual'], costs['Rule-Based'], costs['LLM']],
               marker_color=[colors['Manual'], colors['Rule-Based'], colors['LLM']],
               name="Total Cost"),
        row=1, col=1
    )

    # 2. Histogram: Adjustments
    fig_int.add_trace(
        go.Histogram(x=df['llm_pct'], name="LLM Adjust (%)", marker_color=colors['LLM'], opacity=0.7),
        row=1, col=2
    )
    fig_int.add_trace(
        go.Histogram(x=df['rule_pct'], name="Rule Adjust (%)", marker_color=colors['Rule-Based'], opacity=0.7),
        row=1, col=2
    )

    # 3. Waterfall: Savings
    total_manual = df['annual_holding_cost'].sum()
    total_llm = df['llm_holding_cost'].sum()
    fig_int.add_trace(
        go.Waterfall(
            name="Savings", orientation="v",
            x=["Manual Baseline", "Optimization Impact", "Final LLM Cost"],
            textposition="outside",
            text=[f"${total_manual:,.0f}", f"-${(total_manual-total_llm):,.0f}", f"${total_llm:,.0f}"],
            y=[total_manual, total_llm - total_manual, total_llm],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ),
        row=2, col=1
    )

    # 4. Scatter: Detail Matrix
    fig_int.add_trace(
        go.Scatter(
            x=df['unit_cost'], y=df['llm_pct'],
            mode='markers',
            marker=dict(size=df['safety_stock']/10, color=df['llm_value'], colorscale='Viridis', showscale=True),
            text=df['sku_id'] + ": " + df['description'],
            name="SKU Detail"
        ),
        row=2, col=2
    )

    fig_int.update_layout(height=1000, width=1200, title_text="Safety Stock Optimization Interactive Insights", showlegend=False)
    
    interactive_output = f'{output_dir}/interactive_dashboard.html'
    fig_int.write_html(interactive_output)
    print(f"Interactive dashboard saved to {interactive_output}")

if __name__ == "__main__":
    generate_visuals()
