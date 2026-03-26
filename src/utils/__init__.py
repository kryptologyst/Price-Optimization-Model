"""Utility functions for price optimization model."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved configuration to {config_path}")


def save_model(model: Any, model_path: Union[str, Path]) -> None:
    """Save model to pickle file."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {model_path}")


def load_model(model_path: Union[str, Path]) -> Any:
    """Load model from pickle file."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    return model


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Set seeds for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Set random seeds to {seed}")


def calculate_business_metrics(
    prices: np.ndarray,
    quantities: np.ndarray,
    costs: np.ndarray
) -> Dict[str, float]:
    """Calculate key business metrics."""
    revenues = prices * quantities
    profits = (prices - costs) * quantities
    
    metrics = {
        "total_revenue": np.sum(revenues),
        "total_profit": np.sum(profits),
        "total_quantity": np.sum(quantities),
        "average_price": np.mean(prices),
        "average_margin": np.mean((prices - costs) / prices),
        "profit_margin": np.sum(profits) / np.sum(revenues) if np.sum(revenues) > 0 else 0
    }
    
    return metrics


def validate_data_quality(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate data quality."""
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for negative values where inappropriate
    numeric_columns = df[required_columns].select_dtypes(include=[np.number]).columns
    negative_counts = (df[numeric_columns] < 0).sum()
    if negative_counts.any():
        logger.warning(f"Negative values found: {negative_counts[negative_counts > 0].to_dict()}")
    
    return True


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency."""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for DataFrame."""
    summary = df.describe()
    
    # Add additional statistics
    summary.loc['skewness'] = df.skew()
    summary.loc['kurtosis'] = df.kurtosis()
    summary.loc['missing'] = df.isnull().sum()
    
    return summary


def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Detect outliers using IQR method."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    outliers = z_scores > threshold
    return outliers


def create_time_series_split(
    data: pd.DataFrame,
    time_column: str,
    n_splits: int = 5
) -> list:
    """Create time series splits for cross-validation."""
    from sklearn.model_selection import TimeSeriesSplit
    
    # Sort by time
    data_sorted = data.sort_values(time_column)
    
    # Create splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_idx, test_idx in tscv.split(data_sorted):
        train_data = data_sorted.iloc[train_idx]
        test_data = data_sorted.iloc[test_idx]
        splits.append((train_data, test_data))
    
    return splits


def calculate_feature_correlations(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Calculate correlations between features and target."""
    correlations = df.corr()[target_column].drop(target_column)
    correlations = correlations.sort_values(key=abs, ascending=False)
    return correlations


def create_feature_interactions(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create interaction features between specified columns."""
    df_interactions = df.copy()
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            interaction_name = f"{col1}_x_{col2}"
            df_interactions[interaction_name] = df[col1] * df[col2]
    
    return df_interactions


def normalize_features(df: pd.DataFrame, columns: list, method: str = "standard") -> pd.DataFrame:
    """Normalize specified columns."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    df_normalized = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    df_normalized[columns] = scaler.fit_transform(df[columns])
    
    return df_normalized, scaler


def create_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    """Create lagged features."""
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df_lagged[lag_col] = df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df: pd.DataFrame, columns: list, windows: list) -> pd.DataFrame:
    """Create rolling window features."""
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            # Rolling mean
            mean_col = f"{col}_rolling_mean_{window}"
            df_rolling[mean_col] = df[col].rolling(window=window).mean()
            
            # Rolling std
            std_col = f"{col}_rolling_std_{window}"
            df_rolling[std_col] = df[col].rolling(window=window).std()
            
            # Rolling min/max
            min_col = f"{col}_rolling_min_{window}"
            max_col = f"{col}_rolling_max_{window}"
            df_rolling[min_col] = df[col].rolling(window=window).min()
            df_rolling[max_col] = df[col].rolling(window=window).max()
    
    return df_rolling


def calculate_price_elasticity(prices: np.ndarray, quantities: np.ndarray) -> float:
    """Calculate price elasticity of demand."""
    if len(prices) < 2 or len(quantities) < 2:
        return 0.0
    
    # Use log-log regression
    log_prices = np.log(prices)
    log_quantities = np.log(quantities)
    
    # Simple linear regression
    n = len(log_prices)
    sum_x = np.sum(log_prices)
    sum_y = np.sum(log_quantities)
    sum_xy = np.sum(log_prices * log_quantities)
    sum_x2 = np.sum(log_prices ** 2)
    
    if n * sum_x2 - sum_x ** 2 == 0:
        return 0.0
    
    elasticity = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    return elasticity


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for data."""
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, n - 1)
    
    margin_error = t_val * std_err
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    
    return lower_bound, upper_bound


def bootstrap_metric(data: np.ndarray, metric_func, n_bootstrap: int = 1000) -> dict:
    """Bootstrap a metric to get confidence intervals."""
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_values.append(metric_func(bootstrap_sample))
    
    bootstrap_values = np.array(bootstrap_values)
    
    return {
        "mean": np.mean(bootstrap_values),
        "std": np.std(bootstrap_values),
        "ci_lower": np.percentile(bootstrap_values, 2.5),
        "ci_upper": np.percentile(bootstrap_values, 97.5)
    }
