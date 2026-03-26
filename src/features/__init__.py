"""Feature engineering for price optimization models."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class PriceOptimizationFeatureEngineer:
    """Feature engineering for price optimization models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        
    def create_price_elasticity_features(
        self, 
        transactions: pd.DataFrame, 
        catalog: pd.DataFrame
    ) -> pd.DataFrame:
        """Create price elasticity features."""
        logger.info("Creating price elasticity features...")
        
        # Merge with catalog to get base prices and elasticity
        df = transactions.merge(
            catalog[["product_id", "base_price", "elasticity", "category"]], 
            on="product_id"
        )
        
        # Price ratio features
        df["price_ratio"] = df["price"] / df["base_price"]
        df["price_deviation"] = df["price"] - df["base_price"]
        df["price_deviation_pct"] = (df["price"] - df["base_price"]) / df["base_price"]
        
        # Elasticity-weighted features
        df["elasticity_adjusted_price"] = df["price"] * df["elasticity"]
        df["demand_sensitivity"] = df["price_ratio"] ** df["elasticity"]
        
        # Revenue and profit features
        df["revenue"] = df["price"] * df["quantity"]
        df["profit"] = (df["price"] - df["cost"]) * df["quantity"]
        df["profit_margin"] = (df["price"] - df["cost"]) / df["price"]
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        logger.info("Creating temporal features...")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Time-based features
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        
        # Cyclical encoding
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Weekend indicator
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        return df
    
    def create_customer_features(
        self, 
        df: pd.DataFrame, 
        segments: pd.DataFrame
    ) -> pd.DataFrame:
        """Create customer segment features."""
        logger.info("Creating customer features...")
        
        # Merge with segments
        df = df.merge(segments, on="segment_id", suffixes=("", "_segment"))
        
        # Customer behavior features
        customer_stats = df.groupby("customer_id").agg({
            "quantity": ["mean", "std", "sum"],
            "price": ["mean", "std"],
            "revenue": ["sum", "mean"],
            "transaction_id": "count"
        }).round(3)
        
        customer_stats.columns = [
            "avg_quantity", "std_quantity", "total_quantity",
            "avg_price", "std_price", "total_revenue", "avg_revenue",
            "transaction_count"
        ]
        
        df = df.merge(customer_stats, on="customer_id")
        
        # Customer segment interactions
        df["price_sensitivity_interaction"] = df["price_sensitivity"] * df["price_ratio"]
        df["volume_interaction"] = df["volume_multiplier"] * df["quantity"]
        
        return df
    
    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-level features."""
        logger.info("Creating product features...")
        
        # Product performance features
        product_stats = df.groupby("product_id").agg({
            "quantity": ["mean", "std", "sum"],
            "price": ["mean", "std", "min", "max"],
            "revenue": ["sum", "mean"],
            "transaction_id": "count"
        }).round(3)
        
        product_stats.columns = [
            "prod_avg_quantity", "prod_std_quantity", "prod_total_quantity",
            "prod_avg_price", "prod_std_price", "prod_min_price", "prod_max_price",
            "prod_total_revenue", "prod_avg_revenue", "prod_transaction_count"
        ]
        
        df = df.merge(product_stats, on="product_id")
        
        # Price range features
        df["price_range"] = df["prod_max_price"] - df["prod_min_price"]
        df["price_variability"] = df["prod_std_price"] / df["prod_avg_price"]
        
        # Category encoding
        if "category" not in df.columns:
            logger.warning("Category column not found, skipping category features")
        else:
            if "category_encoder" not in self.encoders:
                self.encoders["category_encoder"] = LabelEncoder()
                df["category_encoded"] = self.encoders["category_encoder"].fit_transform(df["category"])
            else:
                df["category_encoded"] = self.encoders["category_encoder"].transform(df["category"])
        
        return df
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-level features."""
        logger.info("Creating market features...")
        
        # Market-wide statistics
        market_stats = df.groupby("timestamp").agg({
            "quantity": "sum",
            "revenue": "sum",
            "price": "mean",
            "transaction_id": "count"
        }).round(3)
        
        market_stats.columns = ["market_total_quantity", "market_total_revenue", "market_avg_price", "market_transaction_count"]
        
        df = df.merge(market_stats, on="timestamp")
        
        # Relative performance features
        df["quantity_market_share"] = df["quantity"] / df["market_total_quantity"]
        df["revenue_market_share"] = df["revenue"] / df["market_total_revenue"]
        df["price_vs_market"] = df["price"] / df["market_avg_price"]
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Create lagged features."""
        logger.info(f"Creating lag features with lags: {lags}")
        
        df = df.sort_values(["product_id", "timestamp"])
        
        for lag in lags:
            df[f"quantity_lag_{lag}"] = df.groupby("product_id")["quantity"].shift(lag)
            df[f"price_lag_{lag}"] = df.groupby("product_id")["price"].shift(lag)
            df[f"revenue_lag_{lag}"] = df.groupby("product_id")["revenue"].shift(lag)
            
            # Rolling averages
            df[f"quantity_ma_{lag}"] = df.groupby("product_id")["quantity"].rolling(window=lag).mean().reset_index(0, drop=True)
            df[f"price_ma_{lag}"] = df.groupby("product_id")["price"].rolling(window=lag).mean().reset_index(0, drop=True)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features...")
        
        # Price-quantity interactions
        df["price_quantity_interaction"] = df["price"] * df["quantity"]
        df["price_elasticity_interaction"] = df["price"] * df["elasticity"]
        
        # Customer-product interactions
        df["customer_price_sensitivity"] = df["price_sensitivity"] * df["price"]
        df["segment_elasticity_interaction"] = df["price_sensitivity"] * df["elasticity"]
        
        # Temporal interactions
        df["weekend_price_interaction"] = df["is_weekend"] * df["price"]
        df["seasonal_price_interaction"] = df["month_sin"] * df["price"]
        
        return df
    
    def engineer_features(
        self, 
        transactions: pd.DataFrame, 
        catalog: pd.DataFrame, 
        segments: pd.DataFrame,
        include_lags: bool = True,
        lags: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """Engineer all features for price optimization."""
        logger.info("Starting feature engineering...")
        
        # Start with price elasticity features
        df = self.create_price_elasticity_features(transactions, catalog)
        
        # Add temporal features
        df = self.create_temporal_features(df)
        
        # Add customer features
        df = self.create_customer_features(df, segments)
        
        # Add product features
        df = self.create_product_features(df)
        
        # Add market features
        df = self.create_market_features(df)
        
        # Add lag features if requested
        if include_lags:
            df = self.create_lag_features(df, lags)
        
        # Add interaction features
        df = self.create_interaction_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in [
            "transaction_id", "customer_id", "product_id", "timestamp", 
            "segment_id", "name", "category"
        ]]
        
        logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features.")
        
        return df
    
    def prepare_model_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = "quantity",
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for model training."""
        logger.info("Preparing model features...")
        
        # Select features
        feature_cols = [col for col in self.feature_names if col != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].values
        
        # Scale features if requested
        if scale_features:
            if "feature_scaler" not in self.scalers:
                self.scalers["feature_scaler"] = StandardScaler()
                X_scaled = self.scalers["feature_scaler"].fit_transform(X)
            else:
                X_scaled = self.scalers["feature_scaler"].transform(X)
        else:
            X_scaled = X.values
        
        logger.info(f"Prepared {X_scaled.shape[1]} features for {len(y)} samples")
        
        return X_scaled, y, feature_cols
