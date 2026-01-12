# Dynamic Safety Stock Optimizer

This project provides an automated system for optimizing inventory safety stock levels. It evaluates and compares traditional statistical models, heuristic industry rules, and Generative AI reasoning to minimize holding costs while maintaining targeted service levels.

## Methodology

The system evaluates three distinct optimization strategies to provide a comprehensive view of safety stock requirements:

- **Statistical Baseline**: Utilizes standard Z-score calculations based on historical demand volatility and lead time variability. This provides a consistent mathematical foundation.
- **Heuristic Engine**: Implements industry-standard decision rules to account for common supply chain scenarios that fixed formulas may overlook.
- **AI-Driven Optimization**: Leverages Google Gemini to interpret complex demand signals—including seasonality, trends, and erratic patterns—providing more nuanced adjustments than traditional methods.

## Technical Architecture

The core of the system is built with a focus on scalability and resilience:

- **Asynchronous Processing**: The AI optimization layer uses a `DynamicBatchManager` to handle concurrent API requests efficiently.
- **Self-Healing Logic**: To ensure reliability under API constraints, the system automatically adapts batch sizes and falls back to rule-based logic when rate limits are encountered.
- **Comparative Benchmarking**: A centralized benchmarking suite aggregates results across all methods to quantify business impact and inventory value savings.

## File Overview

- `calculator_statistical.py`: Core logic for demand distribution and statistical safety stock.
- `optimizer_heuristic.py`: Rule-based adjustment engine for standard supply chain scenarios.
- `optimizer_ai.py`: Asynchronous engine for GenAI-driven optimization and API management.
- `benchmarker_performance.py`: Evaluation suite for cross-method performance analysis.
- `visualizer_dashboard.py`: Reporting suite for executive dashboards and SKU-level insights.

## Usage

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configuration**:
   Add `GEMINI_API_KEY` to a `.env` file in the root directory.
3. **Execution**:
   ```bash
   python3 benchmarker_performance.py
   ```

## Performance Metrics

The optimizer prioritizes the reduction of unnecessary holding costs. By identifying declining trends or stable demand patterns that don't require high buffers, the system provides actionable recommendations for inventory reduction without increasing stockout risk.

---
*Developed for efficient supply chain inventory management.*
