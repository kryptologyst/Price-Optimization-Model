#!/usr/bin/env python3
"""Main training and optimization script for price optimization model."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import PriceOptimizationDataGenerator, PriceOptimizationConfig
from features import PriceOptimizationFeatureEngineer
from models import create_model_ensemble, PriceOptimizationModel
from optimization import PriceOptimizationWithConstraints, SensitivityAnalyzer
from eval import PriceOptimizationEvaluator, BusinessMetricsCalculator
from viz import PriceOptimizationVisualizer
from utils import setup_logging, load_config, save_model, set_random_seeds


def main():
    """Main training and optimization pipeline."""
    parser = argparse.ArgumentParser(description="Train and optimize price optimization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="assets/models",
                       help="Output directory for models and results")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                       help="Data directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    set_random_seeds(config["data"]["random_seed"])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting price optimization model training and optimization")
    
    # Step 1: Generate or load data
    logger.info("Step 1: Data preparation")
    data_config = config["data"]
    
    if Path(args.data_dir).exists() and any(Path(args.data_dir).glob("*.csv")):
        logger.info("Loading existing data")
        # Load existing data
        data = {}
        for file_path in Path(args.data_dir).glob("*.csv"):
            name = file_path.stem
            data[name] = pd.read_csv(file_path)
    else:
        logger.info("Generating new data")
        # Generate new data
        data_config_obj = PriceOptimizationConfig(**data_config)
        generator = PriceOptimizationDataGenerator(data_config_obj)
        data = generator.generate_all_data()
        
        # Save data
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        generator.save_data(data, data_dir)
    
    logger.info(f"Data loaded: {len(data['catalog'])} products, {len(data['transactions'])} transactions")
    
    # Step 2: Feature engineering
    logger.info("Step 2: Feature engineering")
    feature_engineer = PriceOptimizationFeatureEngineer()
    features_df = feature_engineer.engineer_features(
        data["transactions"], 
        data["catalog"], 
        data["segments"],
        include_lags=True,
        lags=[1, 7, 30]
    )
    
    logger.info(f"Features engineered: {features_df.shape[1]} features, {features_df.shape[0]} samples")
    
    # Step 3: Prepare training data
    logger.info("Step 3: Preparing training data")
    X, y, feature_names = feature_engineer.prepare_model_features(features_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train models
    logger.info("Step 4: Training models")
    ensemble = create_model_ensemble()
    ensemble.fit(X_train, y_train, feature_names)
    
    # Evaluate on test set
    evaluator = PriceOptimizationEvaluator()
    test_metrics = evaluator.evaluate_demand_forecasting(
        ensemble, X_test, y_test, "ensemble"
    )
    
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Step 5: Price optimization
    logger.info("Step 5: Price optimization")
    
    # Create optimizer
    optimizer = PriceOptimizationModel(ensemble)
    optimizer.set_cost_data(data["catalog"])
    
    # Sample products for optimization
    sample_products = data["catalog"].head(10)
    product_features = {}
    
    for _, product in sample_products.iterrows():
        # Use average features for the product
        product_transactions = features_df[features_df["product_id"] == product["product_id"]]
        if len(product_transactions) > 0:
            # Get average features (excluding target and metadata columns)
            feature_cols = [col for col in feature_names if col not in ["quantity", "price"]]
            avg_features = product_transactions[feature_cols].mean().values
            product_features[product["product_id"]] = avg_features
    
    # Optimize prices
    optimization_results = optimizer.batch_optimize(
        product_features, 
        objective=config["optimization"]["objective"]
    )
    
    logger.info("Price optimization completed")
    logger.info(f"Average optimal price: ${optimization_results['optimal_price'].mean():.2f}")
    logger.info(f"Total predicted revenue: ${optimization_results['predicted_revenue'].sum():.2f}")
    logger.info(f"Total predicted profit: ${optimization_results['predicted_profit'].sum():.2f}")
    
    # Step 6: Constraint optimization
    if config["optimization"].get("constraints"):
        logger.info("Step 6: Constraint optimization")
        
        constraint_optimizer = PriceOptimizationWithConstraints(ensemble, data["catalog"])
        
        # Add constraints
        constraints_config = config["optimization"]["constraints"]
        if "min_margin" in constraints_config:
            constraint_optimizer.add_margin_constraint(constraints_config["min_margin"])
        if "min_revenue" in constraints_config:
            constraint_optimizer.add_revenue_constraint(constraints_config["min_revenue"])
        if "min_market_share" in constraints_config:
            constraint_optimizer.add_market_share_constraint(constraints_config["min_market_share"])
        
        # Optimize with constraints
        constraint_result = constraint_optimizer.optimize_prices(
            config["optimization"]["objective"]
        )
        
        if constraint_result.get("success", False):
            logger.info("Constraint optimization successful")
            logger.info(f"Constrained total revenue: ${constraint_result['total_revenue']:,.2f}")
            logger.info(f"Constrained total profit: ${constraint_result['total_profit']:,.2f}")
            logger.info(f"Average margin: {constraint_result['average_margin']*100:.1f}%")
        else:
            logger.warning("Constraint optimization failed")
    
    # Step 7: Sensitivity analysis
    logger.info("Step 7: Sensitivity analysis")
    
    sensitivity_analyzer = SensitivityAnalyzer(constraint_optimizer)
    
    # Analyze price sensitivity
    base_prices = data["catalog"]["base_price"].values[:5]  # Sample first 5 products
    price_sensitivity = sensitivity_analyzer.analyze_price_sensitivity(
        base_prices,
        perturbation_range=tuple(config["optimization"]["sensitivity_analysis"]["perturbation_range"]),
        n_points=config["optimization"]["sensitivity_analysis"]["n_points"]
    )
    
    logger.info("Sensitivity analysis completed")
    
    # Step 8: Business metrics
    logger.info("Step 8: Calculating business metrics")
    
    # Calculate key business metrics
    prices = optimization_results["optimal_price"].values
    demands = optimization_results["predicted_demand"].values
    costs = data["catalog"]["cost"].values[:len(prices)]
    
    business_metrics = BusinessMetricsCalculator()
    
    # Price elasticity
    elasticity = business_metrics.calculate_price_elasticity(prices, demands)
    logger.info(f"Price elasticity: {elasticity:.3f}")
    
    # Market penetration
    market_size = config["business_metrics"]["market_size"]
    total_demand = np.sum(demands)
    penetration = business_metrics.calculate_market_penetration(total_demand, market_size)
    logger.info(f"Market penetration: {penetration:.1f}%")
    
    # Customer lifetime value
    avg_order_value = np.mean(prices)
    purchase_frequency = config["business_metrics"]["purchase_frequency"]
    customer_lifespan = config["business_metrics"]["customer_lifespan"]
    profit_margin = np.mean((prices - costs) / prices)
    
    clv = business_metrics.calculate_customer_lifetime_value(
        avg_order_value, purchase_frequency, customer_lifespan, profit_margin
    )
    logger.info(f"Customer lifetime value: ${clv:.2f}")
    
    # Step 9: Save models and results
    logger.info("Step 9: Saving models and results")
    
    # Save ensemble model
    save_model(ensemble, output_dir / "ensemble_model.pkl")
    
    # Save feature engineer
    save_model(feature_engineer, output_dir / "feature_engineer.pkl")
    
    # Save optimization results
    optimization_results.to_csv(output_dir / "optimization_results.csv", index=False)
    
    # Save evaluation metrics
    evaluation_report = evaluator.generate_evaluation_report()
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(evaluation_report)
    
    # Step 10: Generate visualizations
    logger.info("Step 10: Generating visualizations")
    
    visualizer = PriceOptimizationVisualizer()
    
    # Create sample visualizations
    sample_product = data["catalog"].iloc[0]
    price_range = np.linspace(sample_product["base_price"] * 0.5, sample_product["base_price"] * 2, 50)
    demand_range = 100 * (price_range / sample_product["base_price"]) ** sample_product["elasticity"]
    
    # Price-demand curve
    fig1 = visualizer.plot_price_demand_curve(
        price_range, demand_range,
        optimal_point=(optimization_results.iloc[0]["optimal_price"], 
                     optimization_results.iloc[0]["predicted_demand"])
    )
    
    # Model performance
    y_pred = ensemble.predict(X_test)
    fig2 = visualizer.plot_model_performance(y_test, y_pred, "Ensemble Model")
    
    # Feature importance
    feature_importance = ensemble.get_feature_importance()
    if feature_importance:
        fig3 = visualizer.plot_feature_importance(
            list(feature_importance.keys()),
            list(feature_importance.values()),
            "Feature Importance"
        )
    
    # Save plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    visualizer.save_plots([fig1, fig2, fig3], str(plots_dir))
    
    logger.info("Training and optimization pipeline completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print summary
    print("\n" + "="*80)
    print("PRICE OPTIMIZATION MODEL - TRAINING SUMMARY")
    print("="*80)
    print(f"Products analyzed: {len(sample_products)}")
    print(f"Features engineered: {len(feature_names)}")
    print(f"Model R²: {test_metrics['r2']:.4f}")
    print(f"Average optimal price: ${optimization_results['optimal_price'].mean():.2f}")
    print(f"Total predicted revenue: ${optimization_results['predicted_revenue'].sum():.2f}")
    print(f"Total predicted profit: ${optimization_results['predicted_profit'].sum():.2f}")
    print(f"Price elasticity: {elasticity:.3f}")
    print(f"Market penetration: {penetration:.1f}%")
    print(f"Customer lifetime value: ${clv:.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
