"""Comprehensive evaluation framework for price optimization models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class PriceOptimizationEvaluator:
    """Comprehensive evaluator for price optimization models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results: Dict[str, Dict] = {}
        self.baseline_results: Optional[Dict] = None
        
    def evaluate_demand_forecasting(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Evaluate demand forecasting model performance."""
        logger.info(f"Evaluating demand forecasting for {model_name}")
        
        y_pred = model.predict(X_test)
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
            "smape": self._calculate_smape(y_test, y_pred),
            "bias": np.mean(y_pred - y_test),
            "correlation": np.corrcoef(y_test, y_pred)[0, 1]
        }
        
        # Additional business metrics
        metrics.update({
            "demand_accuracy": self._calculate_demand_accuracy(y_test, y_pred),
            "over_forecast_rate": np.mean(y_pred > y_test) * 100,
            "under_forecast_rate": np.mean(y_pred < y_test) * 100
        })
        
        self.results[model_name] = metrics
        return metrics
        
    def evaluate_price_optimization(
        self, 
        optimizer,
        test_data: pd.DataFrame,
        optimization_results: Dict,
        model_name: str = "optimizer"
    ) -> Dict[str, float]:
        """Evaluate price optimization performance."""
        logger.info(f"Evaluating price optimization for {model_name}")
        
        # Calculate business metrics
        metrics = {
            "total_revenue": optimization_results.get("total_revenue", 0),
            "total_profit": optimization_results.get("total_profit", 0),
            "total_demand": optimization_results.get("total_demand", 0),
            "average_margin": optimization_results.get("average_margin", 0),
            "revenue_lift": self._calculate_revenue_lift(test_data, optimization_results),
            "profit_lift": self._calculate_profit_lift(test_data, optimization_results),
            "demand_impact": self._calculate_demand_impact(test_data, optimization_results)
        }
        
        # Price optimization specific metrics
        if "optimal_values" in optimization_results:
            optimal_prices = optimization_results["optimal_values"]
            base_prices = test_data["base_price"].values
            
            metrics.update({
                "price_change_avg": np.mean((optimal_prices - base_prices) / base_prices) * 100,
                "price_change_std": np.std((optimal_prices - base_prices) / base_prices) * 100,
                "price_volatility": np.std(optimal_prices) / np.mean(optimal_prices),
                "feasibility_score": 1.0 if optimization_results.get("success", False) else 0.0
            })
        
        self.results[model_name] = metrics
        return metrics
        
    def evaluate_constraint_satisfaction(
        self, 
        optimization_results: Dict,
        constraints: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate constraint satisfaction."""
        logger.info("Evaluating constraint satisfaction")
        
        metrics = {}
        
        for i, constraint in enumerate(constraints):
            constraint_name = constraint.get("name", f"constraint_{i}")
            
            if constraint["type"] == "margin":
                min_margin = constraint["value"]
                actual_margin = optimization_results.get("average_margin", 0)
                satisfaction = 1.0 if actual_margin >= min_margin else actual_margin / min_margin
                metrics[f"{constraint_name}_satisfaction"] = satisfaction
                
            elif constraint["type"] == "revenue":
                min_revenue = constraint["value"]
                actual_revenue = optimization_results.get("total_revenue", 0)
                satisfaction = 1.0 if actual_revenue >= min_revenue else actual_revenue / min_revenue
                metrics[f"{constraint_name}_satisfaction"] = satisfaction
                
            elif constraint["type"] == "market_share":
                min_share = constraint["value"]
                # Calculate actual market share
                total_demand = optimization_results.get("total_demand", 0)
                market_size = 10000  # Assumed market size
                actual_share = total_demand / market_size
                satisfaction = 1.0 if actual_share >= min_share else actual_share / min_share
                metrics[f"{constraint_name}_satisfaction"] = satisfaction
        
        # Overall constraint satisfaction
        if metrics:
            metrics["overall_constraint_satisfaction"] = np.mean(list(metrics.values()))
        
        return metrics
        
    def cross_validate_time_series(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        logger.info(f"Performing time series cross-validation with {n_splits} splits")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            "mae": [],
            "rmse": [],
            "r2": [],
            "mape": []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            model.fit(X_train, y_train, [f"feature_{i}" for i in range(X.shape[1])])
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
            
            cv_results["mae"].append(mean_absolute_error(y_test, y_pred))
            cv_results["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_results["r2"].append(r2_score(y_test, y_pred))
            cv_results["mape"].append(mean_absolute_percentage_error(y_test, y_pred) * 100)
        
        return cv_results
        
    def create_model_leaderboard(self) -> pd.DataFrame:
        """Create model performance leaderboard."""
        if not self.results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
            
        # Prepare data for leaderboard
        leaderboard_data = []
        
        for model_name, metrics in self.results.items():
            row = {"model": model_name}
            row.update(metrics)
            leaderboard_data.append(row)
            
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric (R2 for forecasting, profit for optimization)
        if "r2" in leaderboard.columns:
            leaderboard = leaderboard.sort_values("r2", ascending=False)
        elif "total_profit" in leaderboard.columns:
            leaderboard = leaderboard.sort_values("total_profit", ascending=False)
            
        return leaderboard
        
    def compare_with_baseline(self, baseline_results: Dict) -> Dict[str, float]:
        """Compare current results with baseline."""
        self.baseline_results = baseline_results
        
        improvements = {}
        
        for model_name, metrics in self.results.items():
            if model_name in baseline_results:
                baseline_metrics = baseline_results[model_name]
                
                for metric_name, current_value in metrics.items():
                    if metric_name in baseline_metrics:
                        baseline_value = baseline_metrics[metric_name]
                        
                        if baseline_value != 0:
                            improvement = ((current_value - baseline_value) / baseline_value) * 100
                            improvements[f"{model_name}_{metric_name}_improvement"] = improvement
                        else:
                            improvements[f"{model_name}_{metric_name}_improvement"] = float('inf')
                            
        return improvements
        
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("PRICE OPTIMIZATION MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        if self.results:
            report.append("MODEL PERFORMANCE SUMMARY")
            report.append("-" * 40)
            
            leaderboard = self.create_model_leaderboard()
            report.append(leaderboard.to_string(index=False))
            report.append("")
            
            # Best performing model
            if not leaderboard.empty:
                best_model = leaderboard.iloc[0]["model"]
                report.append(f"Best performing model: {best_model}")
                report.append("")
                
        # Baseline comparison
        if self.baseline_results:
            report.append("BASELINE COMPARISON")
            report.append("-" * 40)
            
            improvements = self.compare_with_baseline(self.baseline_results)
            for metric, improvement in improvements.items():
                report.append(f"{metric}: {improvement:.2f}%")
            report.append("")
            
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if self.results:
            # Analyze performance patterns
            forecasting_models = [name for name in self.results.keys() if "r2" in self.results[name]]
            optimization_models = [name for name in self.results.keys() if "total_profit" in self.results[name]]
            
            if forecasting_models:
                best_forecasting = max(forecasting_models, key=lambda x: self.results[x].get("r2", 0))
                report.append(f"• Use {best_forecasting} for demand forecasting")
                
            if optimization_models:
                best_optimization = max(optimization_models, key=lambda x: self.results[x].get("total_profit", 0))
                report.append(f"• Use {best_optimization} for price optimization")
                
        report.append("• Monitor model performance regularly")
        report.append("• Validate predictions against actual business outcomes")
        report.append("• Consider ensemble methods for improved robustness")
        
        return "\n".join(report)
        
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(numerator / denominator) * 100
        
    def _calculate_demand_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate demand accuracy within tolerance."""
        tolerance = 0.1  # 10% tolerance
        accurate = np.abs(y_true - y_pred) / y_true <= tolerance
        return np.mean(accurate) * 100
        
    def _calculate_revenue_lift(self, test_data: pd.DataFrame, results: Dict) -> float:
        """Calculate revenue lift from optimization."""
        baseline_revenue = (test_data["price"] * test_data["quantity"]).sum()
        optimized_revenue = results.get("total_revenue", baseline_revenue)
        
        if baseline_revenue > 0:
            return ((optimized_revenue - baseline_revenue) / baseline_revenue) * 100
        return 0.0
        
    def _calculate_profit_lift(self, test_data: pd.DataFrame, results: Dict) -> float:
        """Calculate profit lift from optimization."""
        baseline_profit = ((test_data["price"] - test_data["cost"]) * test_data["quantity"]).sum()
        optimized_profit = results.get("total_profit", baseline_profit)
        
        if baseline_profit > 0:
            return ((optimized_profit - baseline_profit) / baseline_profit) * 100
        return 0.0
        
    def _calculate_demand_impact(self, test_data: pd.DataFrame, results: Dict) -> float:
        """Calculate demand impact from optimization."""
        baseline_demand = test_data["quantity"].sum()
        optimized_demand = results.get("total_demand", baseline_demand)
        
        if baseline_demand > 0:
            return ((optimized_demand - baseline_demand) / baseline_demand) * 100
        return 0.0


class BusinessMetricsCalculator:
    """Calculate business-specific metrics for price optimization."""
    
    @staticmethod
    def calculate_price_elasticity(
        prices: np.ndarray, 
        quantities: np.ndarray
    ) -> float:
        """Calculate price elasticity of demand."""
        if len(prices) < 2 or len(quantities) < 2:
            return 0.0
            
        # Use log-log regression for elasticity
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
        
    @staticmethod
    def calculate_market_penetration(
        total_demand: float, 
        market_size: float
    ) -> float:
        """Calculate market penetration rate."""
        if market_size <= 0:
            return 0.0
        return (total_demand / market_size) * 100
        
    @staticmethod
    def calculate_customer_lifetime_value(
        avg_order_value: float,
        purchase_frequency: float,
        customer_lifespan: float,
        profit_margin: float
    ) -> float:
        """Calculate customer lifetime value."""
        return avg_order_value * purchase_frequency * customer_lifespan * profit_margin
        
    @staticmethod
    def calculate_inventory_turnover(
        cost_of_goods_sold: float,
        average_inventory: float
    ) -> float:
        """Calculate inventory turnover ratio."""
        if average_inventory <= 0:
            return 0.0
        return cost_of_goods_sold / average_inventory
        
    @staticmethod
    def calculate_gross_margin(
        revenue: float,
        cost_of_goods_sold: float
    ) -> float:
        """Calculate gross margin percentage."""
        if revenue <= 0:
            return 0.0
        return ((revenue - cost_of_goods_sold) / revenue) * 100


class A/BTestEvaluator:
    """Evaluate A/B test results for price optimization."""
    
    def __init__(self):
        """Initialize A/B test evaluator."""
        self.test_results: Dict[str, Dict] = {}
        
    def run_statistical_test(
        self, 
        control_group: np.ndarray, 
        treatment_group: np.ndarray,
        test_name: str = "price_test"
    ) -> Dict[str, float]:
        """Run statistical test comparing control and treatment groups."""
        from scipy import stats
        
        # Calculate basic statistics
        control_mean = np.mean(control_group)
        treatment_mean = np.mean(treatment_group)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_group) - 1) * np.var(control_group) + 
                             (len(treatment_group) - 1) * np.var(treatment_group)) / 
                            (len(control_group) + len(treatment_group) - 2))
        
        if pooled_std > 0:
            cohens_d = (treatment_mean - control_mean) / pooled_std
        else:
            cohens_d = 0.0
            
        # Calculate lift
        lift = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0
        
        results = {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "lift_percent": lift,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        }
        
        self.test_results[test_name] = results
        return results
        
    def calculate_sample_size(
        self, 
        effect_size: float, 
        power: float = 0.8, 
        alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for A/B test."""
        from scipy import stats
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = (2 * (z_alpha + z_beta) ** 2) / (effect_size ** 2)
        return int(np.ceil(n))
        
    def generate_ab_test_report(self, test_name: str) -> str:
        """Generate A/B test report."""
        if test_name not in self.test_results:
            return f"No results found for test: {test_name}"
            
        results = self.test_results[test_name]
        
        report = []
        report.append(f"A/B Test Report: {test_name}")
        report.append("=" * 50)
        report.append(f"Control Group Mean: {results['control_mean']:.2f}")
        report.append(f"Treatment Group Mean: {results['treatment_mean']:.2f}")
        report.append(f"Lift: {results['lift_percent']:.2f}%")
        report.append(f"P-value: {results['p_value']:.4f}")
        report.append(f"Significant: {'Yes' if results['significant'] else 'No'}")
        report.append(f"Effect Size: {results['effect_size']}")
        
        return "\n".join(report)
