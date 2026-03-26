"""Data handling and generation for price optimization model."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PriceOptimizationConfig(BaseModel):
    """Configuration for price optimization data generation."""
    
    n_products: int = Field(default=50, description="Number of products")
    n_customers: int = Field(default=1000, description="Number of customers")
    n_periods: int = Field(default=365, description="Number of time periods")
    price_range: Tuple[float, float] = Field(default=(5.0, 100.0), description="Price range")
    elasticity_range: Tuple[float, float] = Field(default=(-3.0, -0.5), description="Price elasticity range")
    cost_margin: float = Field(default=0.3, description="Cost as fraction of price")
    seasonality_strength: float = Field(default=0.2, description="Seasonal variation strength")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class ProductCatalog(BaseModel):
    """Product catalog schema."""
    
    product_id: str
    name: str
    category: str
    base_price: float
    cost: float
    elasticity: float
    seasonal_factor: float


class CustomerSegment(BaseModel):
    """Customer segment schema."""
    
    segment_id: str
    name: str
    price_sensitivity: float
    volume_multiplier: float
    size: int


class Transaction(BaseModel):
    """Transaction schema."""
    
    transaction_id: str
    customer_id: str
    product_id: str
    price: float
    quantity: int
    timestamp: pd.Timestamp
    segment_id: str


class PriceOptimizationDataGenerator:
    """Generate synthetic data for price optimization modeling."""
    
    def __init__(self, config: PriceOptimizationConfig):
        """Initialize data generator with configuration."""
        self.config = config
        self._set_seeds()
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
    def generate_product_catalog(self) -> pd.DataFrame:
        """Generate synthetic product catalog."""
        logger.info("Generating product catalog...")
        
        categories = ["Electronics", "Clothing", "Home", "Sports", "Books", "Beauty"]
        products = []
        
        for i in range(self.config.n_products):
            category = np.random.choice(categories)
            base_price = np.random.uniform(*self.config.price_range)
            cost = base_price * self.config.cost_margin
            elasticity = np.random.uniform(*self.config.elasticity_range)
            seasonal_factor = np.random.uniform(
                1 - self.config.seasonality_strength,
                1 + self.config.seasonality_strength
            )
            
            products.append({
                "product_id": f"PROD_{i:03d}",
                "name": f"{category} Product {i+1}",
                "category": category,
                "base_price": round(base_price, 2),
                "cost": round(cost, 2),
                "elasticity": round(elasticity, 3),
                "seasonal_factor": round(seasonal_factor, 3)
            })
            
        return pd.DataFrame(products)
    
    def generate_customer_segments(self) -> pd.DataFrame:
        """Generate customer segments."""
        logger.info("Generating customer segments...")
        
        segments = [
            {"name": "Price Sensitive", "price_sensitivity": 1.5, "volume_multiplier": 0.8, "size": 0.3},
            {"name": "Premium", "price_sensitivity": 0.5, "volume_multiplier": 1.2, "size": 0.2},
            {"name": "Regular", "price_sensitivity": 1.0, "volume_multiplier": 1.0, "size": 0.5}
        ]
        
        segment_data = []
        for i, segment in enumerate(segments):
            segment_data.append({
                "segment_id": f"SEG_{i}",
                "name": segment["name"],
                "price_sensitivity": segment["price_sensitivity"],
                "volume_multiplier": segment["volume_multiplier"],
                "size": segment["size"]
            })
            
        return pd.DataFrame(segment_data)
    
    def generate_transactions(
        self, 
        catalog: pd.DataFrame, 
        segments: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate synthetic transaction data."""
        logger.info("Generating transaction data...")
        
        transactions = []
        transaction_id = 0
        
        # Generate customers
        customers = []
        for i, segment in segments.iterrows():
            n_customers_segment = int(self.config.n_customers * segment["size"])
            for j in range(n_customers_segment):
                customers.append({
                    "customer_id": f"CUST_{len(customers):04d}",
                    "segment_id": segment["segment_id"]
                })
        
        # Generate transactions over time
        dates = pd.date_range(
            start="2023-01-01", 
            periods=self.config.n_periods, 
            freq="D"
        )
        
        for date in dates:
            # Daily transaction volume varies
            daily_volume = np.random.poisson(50)
            
            for _ in range(daily_volume):
                customer = np.random.choice(customers)
                product = catalog.sample(1).iloc[0]
                
                # Price varies around base price
                price_variation = np.random.normal(1.0, 0.1)
                price = max(product["base_price"] * price_variation, product["cost"] * 1.1)
                
                # Quantity depends on price elasticity and customer segment
                segment = segments[segments["segment_id"] == customer["segment_id"]].iloc[0]
                base_demand = 10
                
                # Price elasticity effect
                price_ratio = price / product["base_price"]
                elasticity_effect = price_ratio ** product["elasticity"]
                
                # Customer segment effect
                segment_effect = segment["price_sensitivity"] * segment["volume_multiplier"]
                
                # Seasonal effect
                seasonal_effect = product["seasonal_factor"] * (1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365))
                
                quantity = max(1, int(
                    base_demand * elasticity_effect * segment_effect * seasonal_effect
                ))
                
                transactions.append({
                    "transaction_id": f"TXN_{transaction_id:06d}",
                    "customer_id": customer["customer_id"],
                    "product_id": product["product_id"],
                    "price": round(price, 2),
                    "quantity": quantity,
                    "timestamp": date,
                    "segment_id": customer["segment_id"]
                })
                
                transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset."""
        logger.info("Generating complete price optimization dataset...")
        
        catalog = self.generate_product_catalog()
        segments = self.generate_customer_segments()
        transactions = self.generate_transactions(catalog, segments)
        
        return {
            "catalog": catalog,
            "segments": segments,
            "transactions": transactions
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
        """Save generated data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            file_path = output_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {name} data to {file_path}")
    
    def load_data(self, data_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load data from files."""
        data = {}
        
        for file_path in data_dir.glob("*.csv"):
            name = file_path.stem
            data[name] = pd.read_csv(file_path)
            logger.info(f"Loaded {name} data from {file_path}")
        
        return data


def create_sample_data(output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Create sample data for demonstration."""
    if output_dir is None:
        output_dir = Path("data/raw")
    
    config = PriceOptimizationConfig(
        n_products=20,
        n_customers=500,
        n_periods=90,
        random_seed=42
    )
    
    generator = PriceOptimizationDataGenerator(config)
    data = generator.generate_all_data()
    generator.save_data(data, output_dir)
    
    return data
