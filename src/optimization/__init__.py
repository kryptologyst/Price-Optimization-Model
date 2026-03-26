"""Advanced optimization algorithms for price optimization with constraints."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


class ConstraintOptimizer:
    """Base class for constraint optimization."""
    
    def __init__(self):
        """Initialize constraint optimizer."""
        self.constraints: List[Dict] = []
        self.bounds: Optional[List[Tuple[float, float]]] = None
        
    def add_constraint(self, constraint_type: str, **kwargs) -> None:
        """Add constraint to optimization problem."""
        constraint = {"type": constraint_type, **kwargs}
        self.constraints.append(constraint)
        
    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """Set variable bounds."""
        self.bounds = bounds
        
    def optimize(self, objective_func, initial_guess: np.ndarray) -> Dict:
        """Optimize objective function subject to constraints."""
        raise NotImplementedError("Subclasses must implement optimize method")


class ConvexOptimizer(ConstraintOptimizer):
    """Convex optimization using CVXPY."""
    
    def __init__(self):
        """Initialize convex optimizer."""
        super().__init__()
        self.problem = None
        self.variables = None
        
    def optimize(
        self, 
        objective_func, 
        initial_guess: np.ndarray,
        n_variables: int
    ) -> Dict:
        """Solve convex optimization problem."""
        logger.info("Setting up convex optimization problem")
        
        # Define variables
        self.variables = cp.Variable(n_variables)
        
        # Define objective
        objective = cp.Minimize(objective_func(self.variables))
        
        # Add constraints
        constraints = []
        
        for constraint in self.constraints:
            if constraint["type"] == "bound":
                constraints.append(self.variables >= constraint["lower"])
                constraints.append(self.variables <= constraint["upper"])
            elif constraint["type"] == "linear":
                A = constraint["A"]
                b = constraint["b"]
                constraints.append(A @ self.variables <= b)
            elif constraint["type"] == "equality":
                A = constraint["A"]
                b = constraint["b"]
                constraints.append(A @ self.variables == b)
                
        # Create and solve problem
        self.problem = cp.Problem(objective, constraints)
        self.problem.solve(verbose=True)
        
        if self.problem.status == cp.OPTIMAL:
            return {
                "success": True,
                "optimal_values": self.variables.value,
                "optimal_objective": self.problem.value,
                "status": "optimal"
            }
        else:
            return {
                "success": False,
                "optimal_values": None,
                "optimal_objective": None,
                "status": self.problem.status
            }


class ScipyOptimizer(ConstraintOptimizer):
    """Non-linear optimization using SciPy."""
    
    def __init__(self, method: str = "SLSQP"):
        """Initialize SciPy optimizer."""
        super().__init__()
        self.method = method
        
    def optimize(
        self, 
        objective_func, 
        initial_guess: np.ndarray
    ) -> Dict:
        """Solve optimization problem using SciPy."""
        logger.info(f"Solving optimization problem using {self.method}")
        
        # Prepare constraints
        constraints = []
        
        for constraint in self.constraints:
            if constraint["type"] == "inequality":
                constraints.append({
                    "type": "ineq",
                    "fun": constraint["fun"]
                })
            elif constraint["type"] == "equality":
                constraints.append({
                    "type": "eq",
                    "fun": constraint["fun"]
                })
                
        # Solve optimization problem
        result = minimize(
            objective_func,
            initial_guess,
            method=self.method,
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )
        
        return {
            "success": result.success,
            "optimal_values": result.x,
            "optimal_objective": result.fun,
            "status": result.message,
            "iterations": result.nit
        }


class PriceOptimizationWithConstraints:
    """Price optimization with business constraints."""
    
    def __init__(self, demand_model, cost_data: pd.DataFrame):
        """Initialize price optimization with constraints."""
        self.demand_model = demand_model
        self.cost_data = cost_data
        self.optimizer = ConvexOptimizer()
        
    def add_price_bounds(self, min_price: float, max_price: float) -> None:
        """Add price bounds constraint."""
        self.optimizer.add_constraint(
            "bound",
            lower=min_price,
            upper=max_price
        )
        
    def add_margin_constraint(self, min_margin: float) -> None:
        """Add minimum margin constraint."""
        def margin_constraint(prices):
            margins = []
            for i, product_id in enumerate(self.cost_data["product_id"]):
                cost = self.cost_data[self.cost_data["product_id"] == product_id]["cost"].iloc[0]
                margin = (prices[i] - cost) / prices[i]
                margins.append(margin)
            return np.array(margins) - min_margin
            
        self.optimizer.add_constraint("inequality", fun=margin_constraint)
        
    def add_revenue_constraint(self, min_revenue: float) -> None:
        """Add minimum revenue constraint."""
        def revenue_constraint(prices):
            total_revenue = 0
            for i, product_id in enumerate(self.cost_data["product_id"]):
                # Simplified revenue calculation
                # In practice, you'd use the demand model
                base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
                elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
                
                # Estimate demand using elasticity
                price_ratio = prices[i] / base_price
                demand = 100 * (price_ratio ** elasticity)  # Simplified
                revenue = prices[i] * demand
                total_revenue += revenue
                
            return total_revenue - min_revenue
            
        self.optimizer.add_constraint("inequality", fun=revenue_constraint)
        
    def add_market_share_constraint(self, min_market_share: float) -> None:
        """Add minimum market share constraint."""
        def market_share_constraint(prices):
            # Simplified market share calculation
            total_demand = 0
            for i, product_id in enumerate(self.cost_data["product_id"]):
                base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
                elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
                
                price_ratio = prices[i] / base_price
                demand = 100 * (price_ratio ** elasticity)
                total_demand += demand
                
            # Assume total market size
            market_size = 10000
            market_share = total_demand / market_size
            
            return market_share - min_market_share
            
        self.optimizer.add_constraint("inequality", fun=market_share_constraint)
        
    def optimize_prices(
        self, 
        objective: str = "profit",
        initial_prices: Optional[np.ndarray] = None
    ) -> Dict:
        """Optimize prices subject to constraints."""
        n_products = len(self.cost_data)
        
        if initial_prices is None:
            initial_prices = self.cost_data["base_price"].values
            
        # Define objective function
        if objective == "profit":
            def objective_func(prices):
                total_profit = 0
                for i, product_id in enumerate(self.cost_data["product_id"]):
                    cost = self.cost_data[self.cost_data["product_id"] == product_id]["cost"].iloc[0]
                    base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
                    elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
                    
                    # Estimate demand
                    price_ratio = prices[i] / base_price
                    demand = 100 * (price_ratio ** elasticity)
                    
                    profit = (prices[i] - cost) * demand
                    total_profit += profit
                    
                return -total_profit  # Minimize negative profit
                
        elif objective == "revenue":
            def objective_func(prices):
                total_revenue = 0
                for i, product_id in enumerate(self.cost_data["product_id"]):
                    base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
                    elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
                    
                    price_ratio = prices[i] / base_price
                    demand = 100 * (price_ratio ** elasticity)
                    
                    revenue = prices[i] * demand
                    total_revenue += revenue
                    
                return -total_revenue  # Minimize negative revenue
        else:
            raise ValueError(f"Unknown objective: {objective}")
            
        # Set bounds
        bounds = [(cost * 1.1, cost * 5.0) for cost in self.cost_data["cost"].values]
        self.optimizer.set_bounds(bounds)
        
        # Solve optimization problem
        result = self.optimizer.optimize(objective_func, initial_prices, n_products)
        
        if result["success"]:
            # Calculate additional metrics
            optimal_prices = result["optimal_values"]
            metrics = self._calculate_metrics(optimal_prices, objective)
            result.update(metrics)
            
        return result
        
    def _calculate_metrics(self, prices: np.ndarray, objective: str) -> Dict:
        """Calculate optimization metrics."""
        metrics = {}
        
        total_revenue = 0
        total_profit = 0
        total_demand = 0
        
        for i, product_id in enumerate(self.cost_data["product_id"]):
            cost = self.cost_data[self.cost_data["product_id"] == product_id]["cost"].iloc[0]
            base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
            elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
            
            price_ratio = prices[i] / base_price
            demand = 100 * (price_ratio ** elasticity)
            
            revenue = prices[i] * demand
            profit = (prices[i] - cost) * demand
            
            total_revenue += revenue
            total_profit += profit
            total_demand += demand
            
        metrics["total_revenue"] = total_revenue
        metrics["total_profit"] = total_profit
        metrics["total_demand"] = total_demand
        metrics["average_margin"] = np.mean([(p - c) / p for p, c in zip(prices, self.cost_data["cost"].values)])
        
        return metrics


class MultiObjectiveOptimizer:
    """Multi-objective optimization for price optimization."""
    
    def __init__(self, demand_model, cost_data: pd.DataFrame):
        """Initialize multi-objective optimizer."""
        self.demand_model = demand_model
        self.cost_data = cost_data
        
    def optimize_pareto_frontier(
        self, 
        objectives: List[str],
        n_points: int = 50
    ) -> pd.DataFrame:
        """Find Pareto frontier for multiple objectives."""
        logger.info(f"Finding Pareto frontier for objectives: {objectives}")
        
        # Generate random price combinations
        results = []
        
        for _ in range(n_points * 10):  # Generate more points than needed
            # Random prices within bounds
            prices = np.random.uniform(
                self.cost_data["cost"].values * 1.1,
                self.cost_data["cost"].values * 5.0
            )
            
            # Calculate objectives
            metrics = self._calculate_objectives(prices, objectives)
            
            # Add price information
            metrics.update({
                f"price_{i}": prices[i] 
                for i in range(len(prices))
            })
            
            results.append(metrics)
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Find Pareto optimal points
        pareto_points = self._find_pareto_optimal(df, objectives)
        
        return pareto_points
        
    def _calculate_objectives(self, prices: np.ndarray, objectives: List[str]) -> Dict:
        """Calculate objective values for given prices."""
        metrics = {}
        
        total_revenue = 0
        total_profit = 0
        total_demand = 0
        min_margin = float('inf')
        
        for i, product_id in enumerate(self.cost_data["product_id"]):
            cost = self.cost_data[self.cost_data["product_id"] == product_id]["cost"].iloc[0]
            base_price = self.cost_data[self.cost_data["product_id"] == product_id]["base_price"].iloc[0]
            elasticity = self.cost_data[self.cost_data["product_id"] == product_id]["elasticity"].iloc[0]
            
            price_ratio = prices[i] / base_price
            demand = 100 * (price_ratio ** elasticity)
            
            revenue = prices[i] * demand
            profit = (prices[i] - cost) * demand
            margin = (prices[i] - cost) / prices[i]
            
            total_revenue += revenue
            total_profit += profit
            total_demand += demand
            min_margin = min(min_margin, margin)
            
        # Calculate objectives
        if "revenue" in objectives:
            metrics["revenue"] = total_revenue
        if "profit" in objectives:
            metrics["profit"] = total_profit
        if "demand" in objectives:
            metrics["demand"] = total_demand
        if "margin" in objectives:
            metrics["min_margin"] = min_margin
            
        return metrics
        
    def _find_pareto_optimal(self, df: pd.DataFrame, objectives: List[str]) -> pd.DataFrame:
        """Find Pareto optimal points."""
        pareto_indices = []
        
        for i in range(len(df)):
            is_pareto = True
            
            for j in range(len(df)):
                if i == j:
                    continue
                    
                # Check if point j dominates point i
                dominates = True
                for obj in objectives:
                    if df.iloc[j][obj] <= df.iloc[i][obj]:
                        dominates = False
                        break
                        
                if dominates:
                    is_pareto = False
                    break
                    
            if is_pareto:
                pareto_indices.append(i)
                
        return df.iloc[pareto_indices].reset_index(drop=True)


class SensitivityAnalyzer:
    """Sensitivity analysis for price optimization."""
    
    def __init__(self, optimizer: PriceOptimizationWithConstraints):
        """Initialize sensitivity analyzer."""
        self.optimizer = optimizer
        
    def analyze_price_sensitivity(
        self, 
        base_prices: np.ndarray,
        perturbation_range: Tuple[float, float] = (-0.1, 0.1),
        n_points: int = 21
    ) -> pd.DataFrame:
        """Analyze sensitivity to price changes."""
        logger.info("Analyzing price sensitivity")
        
        results = []
        
        for i, product_id in enumerate(self.optimizer.cost_data["product_id"]):
            base_price = base_prices[i]
            
            # Test different price perturbations
            perturbations = np.linspace(perturbation_range[0], perturbation_range[1], n_points)
            
            for perturbation in perturbations:
                test_prices = base_prices.copy()
                test_prices[i] = base_price * (1 + perturbation)
                
                # Calculate metrics
                metrics = self.optimizer._calculate_metrics(test_prices, "profit")
                
                results.append({
                    "product_id": product_id,
                    "price_change_pct": perturbation * 100,
                    "new_price": test_prices[i],
                    "total_revenue": metrics["total_revenue"],
                    "total_profit": metrics["total_profit"],
                    "total_demand": metrics["total_demand"],
                    "average_margin": metrics["average_margin"]
                })
                
        return pd.DataFrame(results)
        
    def analyze_constraint_sensitivity(
        self, 
        constraint_param: str,
        param_range: Tuple[float, float],
        n_points: int = 21
    ) -> pd.DataFrame:
        """Analyze sensitivity to constraint parameters."""
        logger.info(f"Analyzing sensitivity to {constraint_param}")
        
        results = []
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        
        for param_value in param_values:
            # Create new optimizer with modified constraint
            temp_optimizer = PriceOptimizationWithConstraints(
                self.optimizer.demand_model,
                self.optimizer.cost_data
            )
            
            # Set constraint parameter
            if constraint_param == "min_margin":
                temp_optimizer.add_margin_constraint(param_value)
            elif constraint_param == "min_revenue":
                temp_optimizer.add_revenue_constraint(param_value)
            elif constraint_param == "min_market_share":
                temp_optimizer.add_market_share_constraint(param_value)
            else:
                raise ValueError(f"Unknown constraint parameter: {constraint_param}")
                
            # Optimize
            result = temp_optimizer.optimize_prices("profit")
            
            if result["success"]:
                results.append({
                    constraint_param: param_value,
                    "total_revenue": result["total_revenue"],
                    "total_profit": result["total_profit"],
                    "total_demand": result["total_demand"],
                    "average_margin": result["average_margin"],
                    "feasible": True
                })
            else:
                results.append({
                    constraint_param: param_value,
                    "total_revenue": None,
                    "total_profit": None,
                    "total_demand": None,
                    "average_margin": None,
                    "feasible": False
                })
                
        return pd.DataFrame(results)
