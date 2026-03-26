"""Test suite for price optimization model."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.data import PriceOptimizationDataGenerator, PriceOptimizationConfig
from src.features import PriceOptimizationFeatureEngineer
from src.models import (
    ElasticityModel, RandomForestDemandModel, XGBoostDemandModel,
    ModelEnsemble, PriceOptimizationModel
)
from src.optimization import PriceOptimizationWithConstraints
from src.eval import PriceOptimizationEvaluator, BusinessMetricsCalculator
from src.utils import calculate_business_metrics, validate_data_quality


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = PriceOptimizationConfig(
            n_products=10,
            n_customers=100,
            n_periods=30,
            random_seed=42
        )
        
        assert config.n_products == 10
        assert config.n_customers == 100
        assert config.n_periods == 30
        assert config.random_seed == 42
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        assert generator.config.n_products == 5
        assert generator.config.n_customers == 50
        assert generator.config.n_periods == 10
    
    def test_product_catalog_generation(self):
        """Test product catalog generation."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        catalog = generator.generate_product_catalog()
        
        assert len(catalog) == 5
        assert all(col in catalog.columns for col in [
            "product_id", "name", "category", "base_price", 
            "cost", "elasticity", "seasonal_factor"
        ])
        assert all(catalog["base_price"] > 0)
        assert all(catalog["cost"] > 0)
        assert all(catalog["elasticity"] < 0)  # Negative elasticity
    
    def test_customer_segments_generation(self):
        """Test customer segments generation."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        segments = generator.generate_customer_segments()
        
        assert len(segments) == 3  # Price Sensitive, Premium, Regular
        assert all(col in segments.columns for col in [
            "segment_id", "name", "price_sensitivity", 
            "volume_multiplier", "size"
        ])
        assert abs(segments["size"].sum() - 1.0) < 1e-6  # Should sum to 1
    
    def test_transaction_generation(self):
        """Test transaction generation."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        catalog = generator.generate_product_catalog()
        segments = generator.generate_customer_segments()
        transactions = generator.generate_transactions(catalog, segments)
        
        assert len(transactions) > 0
        assert all(col in transactions.columns for col in [
            "transaction_id", "customer_id", "product_id", 
            "price", "quantity", "timestamp", "segment_id"
        ])
        assert all(transactions["price"] > 0)
        assert all(transactions["quantity"] > 0)


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def setup_method(self):
        """Setup test data."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        self.catalog = generator.generate_product_catalog()
        self.segments = generator.generate_customer_segments()
        self.transactions = generator.generate_transactions(self.catalog, self.segments)
        self.feature_engineer = PriceOptimizationFeatureEngineer()
    
    def test_price_elasticity_features(self):
        """Test price elasticity feature creation."""
        df = self.feature_engineer.create_price_elasticity_features(
            self.transactions, self.catalog
        )
        
        assert "price_ratio" in df.columns
        assert "price_deviation" in df.columns
        assert "revenue" in df.columns
        assert "profit" in df.columns
        assert all(df["price_ratio"] > 0)
    
    def test_temporal_features(self):
        """Test temporal feature creation."""
        df = self.feature_engineer.create_temporal_features(self.transactions)
        
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "day_of_week_sin" in df.columns
        assert "day_of_week_cos" in df.columns
        assert "is_weekend" in df.columns
    
    def test_customer_features(self):
        """Test customer feature creation."""
        df = self.feature_engineer.create_price_elasticity_features(
            self.transactions, self.catalog
        )
        df = self.feature_engineer.create_customer_features(df, self.segments)
        
        assert "price_sensitivity" in df.columns
        assert "volume_multiplier" in df.columns
        assert "avg_quantity" in df.columns
        assert "total_revenue" in df.columns
    
    def test_feature_preparation(self):
        """Test feature preparation for modeling."""
        df = self.feature_engineer.engineer_features(
            self.transactions, self.catalog, self.segments, include_lags=False
        )
        
        X, y, feature_names = self.feature_engineer.prepare_model_features(df)
        
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(feature_names)
        assert len(y) == len(df)
        assert "quantity" not in feature_names  # Should be excluded as target


class TestModels:
    """Test model functionality."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X = np.random.rand(100, 10)
        self.y = np.random.rand(100)
        self.feature_names = [f"feature_{i}" for i in range(10)]
    
    def test_elasticity_model(self):
        """Test elasticity model."""
        model = ElasticityModel()
        model.fit(self.X, self.y, self.feature_names)
        
        assert model.is_fitted
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        assert all(predictions >= 0)  # Non-negative predictions
    
    def test_random_forest_model(self):
        """Test random forest model."""
        model = RandomForestDemandModel()
        model.fit(self.X, self.y, self.feature_names)
        
        assert model.is_fitted
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        importance = model.get_feature_importance()
        assert len(importance) == len(self.feature_names)
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        model = XGBoostDemandModel()
        model.fit(self.X, self.y, self.feature_names)
        
        assert model.is_fitted
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
    
    def test_model_ensemble(self):
        """Test model ensemble."""
        models = [
            ElasticityModel(),
            RandomForestDemandModel(),
            XGBoostDemandModel()
        ]
        ensemble = ModelEnsemble(models)
        ensemble.fit(self.X, self.y, self.feature_names)
        
        assert ensemble.is_fitted
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        
        predictions = ensemble.predict(self.X)
        assert len(predictions) == len(self.y)
    
    def test_price_optimization_model(self):
        """Test price optimization model."""
        demand_model = ElasticityModel()
        demand_model.fit(self.X, self.y, self.feature_names)
        
        optimizer = PriceOptimizationModel(demand_model)
        
        # Create dummy cost data
        cost_data = pd.DataFrame({
            "product_id": ["PROD_001"],
            "cost": [10.0]
        })
        optimizer.set_cost_data(cost_data)
        
        # Test optimization
        features = np.random.rand(5)
        result = optimizer.optimize_price("PROD_001", features)
        
        assert "optimal_price" in result
        assert "predicted_demand" in result
        assert "predicted_revenue" in result
        assert result["optimal_price"] > 0


class TestOptimization:
    """Test optimization functionality."""
    
    def setup_method(self):
        """Setup test data."""
        config = PriceOptimizationConfig(n_products=5, n_customers=50, n_periods=10)
        generator = PriceOptimizationDataGenerator(config)
        
        self.catalog = generator.generate_product_catalog()
        
        # Create dummy demand model
        self.demand_model = Mock()
        self.demand_model.is_fitted = True
        self.demand_model.predict.return_value = np.array([100])
    
    def test_constraint_optimizer_initialization(self):
        """Test constraint optimizer initialization."""
        optimizer = PriceOptimizationWithConstraints(self.demand_model, self.catalog)
        
        assert optimizer.demand_model == self.demand_model
        assert len(optimizer.cost_data) == len(self.catalog)
    
    def test_margin_constraint(self):
        """Test margin constraint addition."""
        optimizer = PriceOptimizationWithConstraints(self.demand_model, self.catalog)
        optimizer.add_margin_constraint(0.2)
        
        assert len(optimizer.optimizer.constraints) == 1
        assert optimizer.optimizer.constraints[0]["type"] == "inequality"
    
    def test_revenue_constraint(self):
        """Test revenue constraint addition."""
        optimizer = PriceOptimizationWithConstraints(self.demand_model, self.catalog)
        optimizer.add_revenue_constraint(10000)
        
        assert len(optimizer.optimizer.constraints) == 1
        assert optimizer.optimizer.constraints[0]["type"] == "inequality"


class TestEvaluation:
    """Test evaluation functionality."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.y_true = np.random.rand(100)
        self.y_pred = np.random.rand(100)
        self.evaluator = PriceOptimizationEvaluator()
    
    def test_demand_forecasting_evaluation(self):
        """Test demand forecasting evaluation."""
        model = Mock()
        model.predict.return_value = self.y_pred
        
        metrics = self.evaluator.evaluate_demand_forecasting(
            model, self.y_true, self.y_pred, "test_model"
        )
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
    
    def test_price_optimization_evaluation(self):
        """Test price optimization evaluation."""
        optimizer = Mock()
        test_data = pd.DataFrame({
            "price": [10, 20, 30],
            "quantity": [100, 80, 60],
            "cost": [5, 10, 15]
        })
        
        optimization_results = {
            "total_revenue": 10000,
            "total_profit": 5000,
            "total_demand": 240,
            "average_margin": 0.5
        }
        
        metrics = self.evaluator.evaluate_price_optimization(
            optimizer, test_data, optimization_results, "test_optimizer"
        )
        
        assert "total_revenue" in metrics
        assert "total_profit" in metrics
        assert "revenue_lift" in metrics
        assert "profit_lift" in metrics
    
    def test_model_leaderboard(self):
        """Test model leaderboard creation."""
        # Add some results
        self.evaluator.results = {
            "model1": {"r2": 0.8, "mae": 10},
            "model2": {"r2": 0.9, "mae": 8}
        }
        
        leaderboard = self.evaluator.create_model_leaderboard()
        
        assert len(leaderboard) == 2
        assert leaderboard.iloc[0]["model"] == "model2"  # Higher R2 should be first


class TestBusinessMetrics:
    """Test business metrics calculation."""
    
    def test_price_elasticity_calculation(self):
        """Test price elasticity calculation."""
        prices = np.array([10, 12, 14, 16, 18])
        quantities = np.array([100, 80, 60, 40, 20])
        
        elasticity = BusinessMetricsCalculator.calculate_price_elasticity(prices, quantities)
        
        assert elasticity < 0  # Should be negative
        assert not np.isnan(elasticity)
    
    def test_market_penetration_calculation(self):
        """Test market penetration calculation."""
        penetration = BusinessMetricsCalculator.calculate_market_penetration(1000, 10000)
        
        assert penetration == 10.0  # 1000/10000 * 100
    
    def test_gross_margin_calculation(self):
        """Test gross margin calculation."""
        margin = BusinessMetricsCalculator.calculate_gross_margin(1000, 600)
        
        assert margin == 40.0  # (1000-600)/1000 * 100


class TestUtils:
    """Test utility functions."""
    
    def test_business_metrics_calculation(self):
        """Test business metrics calculation."""
        prices = np.array([10, 20, 30])
        quantities = np.array([100, 80, 60])
        costs = np.array([5, 10, 15])
        
        metrics = calculate_business_metrics(prices, quantities, costs)
        
        assert "total_revenue" in metrics
        assert "total_profit" in metrics
        assert "average_margin" in metrics
        assert metrics["total_revenue"] == 1000 + 1600 + 1800
        assert metrics["total_profit"] == 500 + 800 + 900
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        df = pd.DataFrame({
            "price": [10, 20, 30],
            "quantity": [100, 80, 60],
            "cost": [5, 10, 15]
        })
        
        required_columns = ["price", "quantity", "cost"]
        is_valid = validate_data_quality(df, required_columns)
        
        assert is_valid
    
    def test_data_quality_validation_missing_columns(self):
        """Test data quality validation with missing columns."""
        df = pd.DataFrame({
            "price": [10, 20, 30],
            "quantity": [100, 80, 60]
        })
        
        required_columns = ["price", "quantity", "cost"]
        is_valid = validate_data_quality(df, required_columns)
        
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__])
