# Price Optimization Model

A comprehensive price optimization model for business operations and analytics, featuring advanced demand forecasting, constraint optimization, and interactive visualizations.

## DISCLAIMER

**IMPORTANT:** This is a research and educational tool for price optimization analysis. It is NOT intended for automated decision-making without human review. All recommendations should be validated by business experts before implementation. This model is experimental and may not reflect real-world market conditions.

## Features

- **Advanced Demand Forecasting**: Multiple ML models including Elasticity, Random Forest, XGBoost, LightGBM, and Gradient Boosting
- **Constraint Optimization**: Business constraints including margin, revenue, and market share requirements
- **Multi-objective Optimization**: Pareto frontier analysis for competing objectives
- **Sensitivity Analysis**: Price and constraint sensitivity evaluation
- **Interactive Dashboard**: Streamlit-based web application for exploration
- **Comprehensive Evaluation**: Business and ML metrics with A/B testing capabilities
- **Rich Visualizations**: Price-demand curves, optimization results, and business dashboards

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Price-Optimization-Model.git
cd Price-Optimization-Model

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Running the Demo

```bash
# Start the Streamlit demo
streamlit run demo/app.py

# Or run the main script
python scripts/train_and_optimize.py
```

### Basic Usage

```python
from src.data import PriceOptimizationDataGenerator, PriceOptimizationConfig
from src.models import create_model_ensemble
from src.optimization import PriceOptimizationWithConstraints

# Generate sample data
config = PriceOptimizationConfig(n_products=20, n_customers=500, n_periods=90)
generator = PriceOptimizationDataGenerator(config)
data = generator.generate_all_data()

# Train models
ensemble = create_model_ensemble()
# ... training code ...

# Optimize prices
optimizer = PriceOptimizationWithConstraints(model, data["catalog"])
optimizer.add_margin_constraint(0.2)
result = optimizer.optimize_prices("profit")
```

## Project Structure

```
price-optimization-model/
├── src/                    # Source code
│   ├── data/              # Data generation and handling
│   ├── features/          # Feature engineering
│   ├── models/            # ML models and ensembles
│   ├── optimization/       # Constraint optimization
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── demo/                  # Streamlit demo application
├── assets/                # Generated plots and models
└── data/                  # Data storage
    ├── raw/               # Raw data files
    ├── processed/         # Processed data files
    └── external/          # External data sources
```

## Data Schema

### Product Catalog
- `product_id`: Unique product identifier
- `name`: Product name
- `category`: Product category
- `base_price`: Base price
- `cost`: Product cost
- `elasticity`: Price elasticity coefficient
- `seasonal_factor`: Seasonal demand factor

### Customer Segments
- `segment_id`: Segment identifier
- `name`: Segment name
- `price_sensitivity`: Price sensitivity multiplier
- `volume_multiplier`: Volume multiplier
- `size`: Segment size (fraction)

### Transactions
- `transaction_id`: Transaction identifier
- `customer_id`: Customer identifier
- `product_id`: Product identifier
- `price`: Transaction price
- `quantity`: Quantity purchased
- `timestamp`: Transaction timestamp
- `segment_id`: Customer segment

## Models

### Demand Forecasting Models

1. **Elasticity Model**: Linear regression with price elasticity
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting with XGBoost
4. **LightGBM**: Light gradient boosting machine
5. **Gradient Boosting**: Scikit-learn gradient boosting
6. **Model Ensemble**: Weighted combination of all models

### Optimization Algorithms

1. **Convex Optimization**: CVXPY-based constraint optimization
2. **Multi-objective Optimization**: Pareto frontier analysis
3. **Sensitivity Analysis**: Parameter sensitivity evaluation

## Evaluation Metrics

### ML Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)
- Symmetric MAPE (SMAPE)

### Business Metrics
- Revenue Lift
- Profit Lift
- Demand Impact
- Price Elasticity
- Market Penetration
- Customer Lifetime Value
- Inventory Turnover
- Gross Margin

## Configuration

The model can be configured using YAML files in the `configs/` directory:

```yaml
# configs/default.yaml
data:
  n_products: 50
  n_customers: 1000
  n_periods: 365

models:
  random_forest:
    n_estimators: 100
    max_depth: null

optimization:
  objective: "profit"
  constraints:
    min_margin: 0.2
    min_revenue: 10000
```

## Demo Application

The Streamlit demo provides an interactive interface for:

1. **Data Overview**: Explore generated datasets
2. **Model Training**: Train and compare different models
3. **Optimization**: Run price optimization with constraints
4. **Evaluation**: View performance metrics and A/B test results
5. **Visualizations**: Interactive charts and dashboards

## API Reference

### Data Generation

```python
from src.data import PriceOptimizationDataGenerator, PriceOptimizationConfig

config = PriceOptimizationConfig(
    n_products=50,
    n_customers=1000,
    n_periods=365,
    random_seed=42
)

generator = PriceOptimizationDataGenerator(config)
data = generator.generate_all_data()
```

### Model Training

```python
from src.models import create_model_ensemble
from src.features import PriceOptimizationFeatureEngineer

# Feature engineering
feature_engineer = PriceOptimizationFeatureEngineer()
features_df = feature_engineer.engineer_features(transactions, catalog, segments)

# Model training
ensemble = create_model_ensemble()
X, y, feature_names = feature_engineer.prepare_model_features(features_df)
ensemble.fit(X, y, feature_names)
```

### Price Optimization

```python
from src.optimization import PriceOptimizationWithConstraints

optimizer = PriceOptimizationWithConstraints(model, catalog)
optimizer.add_margin_constraint(0.2)
optimizer.add_revenue_constraint(10000)

result = optimizer.optimize_prices("profit")
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{price_optimization_model,
  title={Price Optimization Model for Business Operations},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Price-Optimization-Model}
}
```

## Support

For questions and support, please open an issue on GitHub or contact the development team.

## Changelog

### Version 1.0.0
- Initial release
- Basic demand forecasting models
- Constraint optimization
- Streamlit demo application
- Comprehensive evaluation framework
# Price-Optimization-Model
