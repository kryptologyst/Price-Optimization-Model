"""Visualization utilities for price optimization analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PriceOptimizationVisualizer:
    """Comprehensive visualization tools for price optimization."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with style."""
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_price_demand_curve(
        self, 
        prices: np.ndarray, 
        demands: np.ndarray,
        optimal_point: Optional[Tuple[float, float]] = None,
        title: str = "Price-Demand Curve"
    ) -> plt.Figure:
        """Plot price-demand relationship."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(prices, demands, 'b-', linewidth=2, label='Demand Curve')
        
        if optimal_point:
            ax.scatter(*optimal_point, color='red', s=100, zorder=5, label='Optimal Point')
            ax.annotate(
                f'Optimal: ${optimal_point[0]:.2f}',
                xy=optimal_point,
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
        
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Demand')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_revenue_profit_curves(
        self, 
        prices: np.ndarray, 
        revenues: np.ndarray,
        profits: np.ndarray,
        costs: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """Plot revenue and profit curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Revenue curve
        ax1.plot(prices, revenues, 'g-', linewidth=2, label='Revenue')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Revenue ($)')
        ax1.set_title('Revenue Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Profit curve
        ax2.plot(prices, profits, 'r-', linewidth=2, label='Profit')
        if costs is not None:
            ax2.plot(prices, costs, 'b--', linewidth=1, label='Cost')
        ax2.set_xlabel('Price ($)')
        ax2.set_ylabel('Profit ($)')
        ax2.set_title('Profit Curve')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_price_elasticity(
        self, 
        prices: np.ndarray, 
        elasticities: np.ndarray,
        title: str = "Price Elasticity"
    ) -> plt.Figure:
        """Plot price elasticity."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(prices, elasticities, 'purple', linewidth=2, marker='o')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Unit Elasticity')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Price Elasticity')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add elasticity interpretation
        ax.text(0.02, 0.98, 'Elastic (|E| > 1)', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.02, 0.88, 'Inelastic (|E| < 1)', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig
        
    def plot_model_performance(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> plt.Figure:
        """Plot model performance comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{model_name}: Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}: Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Distribution of residuals
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{model_name}: Residual Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title(f'{model_name}: Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        importance_values: List[float],
        title: str = "Feature Importance",
        top_n: int = 20
    ) -> plt.Figure:
        """Plot feature importance."""
        # Sort features by importance
        sorted_features = sorted(zip(feature_names, importance_values), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(importances[i] / max(importances)))
        
        plt.tight_layout()
        return fig
        
    def plot_optimization_results(
        self, 
        results_df: pd.DataFrame,
        title: str = "Optimization Results"
    ) -> plt.Figure:
        """Plot optimization results comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue comparison
        if 'total_revenue' in results_df.columns:
            axes[0, 0].bar(results_df.index, results_df['total_revenue'])
            axes[0, 0].set_title('Total Revenue')
            axes[0, 0].set_ylabel('Revenue ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Profit comparison
        if 'total_profit' in results_df.columns:
            axes[0, 1].bar(results_df.index, results_df['total_profit'])
            axes[0, 1].set_title('Total Profit')
            axes[0, 1].set_ylabel('Profit ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Margin comparison
        if 'average_margin' in results_df.columns:
            axes[1, 0].bar(results_df.index, results_df['average_margin'])
            axes[1, 0].set_title('Average Margin')
            axes[1, 0].set_ylabel('Margin (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Demand comparison
        if 'total_demand' in results_df.columns:
            axes[1, 1].bar(results_df.index, results_df['total_demand'])
            axes[1, 1].set_title('Total Demand')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
        
    def plot_sensitivity_analysis(
        self, 
        sensitivity_df: pd.DataFrame,
        parameter: str,
        title: str = "Sensitivity Analysis"
    ) -> plt.Figure:
        """Plot sensitivity analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue sensitivity
        if 'total_revenue' in sensitivity_df.columns:
            axes[0, 0].plot(sensitivity_df[parameter], sensitivity_df['total_revenue'], 'b-', marker='o')
            axes[0, 0].set_xlabel(parameter)
            axes[0, 0].set_ylabel('Total Revenue')
            axes[0, 0].set_title('Revenue Sensitivity')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Profit sensitivity
        if 'total_profit' in sensitivity_df.columns:
            axes[0, 1].plot(sensitivity_df[parameter], sensitivity_df['total_profit'], 'r-', marker='o')
            axes[0, 1].set_xlabel(parameter)
            axes[0, 1].set_ylabel('Total Profit')
            axes[0, 1].set_title('Profit Sensitivity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Margin sensitivity
        if 'average_margin' in sensitivity_df.columns:
            axes[1, 0].plot(sensitivity_df[parameter], sensitivity_df['average_margin'], 'g-', marker='o')
            axes[1, 0].set_xlabel(parameter)
            axes[0, 0].set_ylabel('Average Margin')
            axes[1, 0].set_title('Margin Sensitivity')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Demand sensitivity
        if 'total_demand' in sensitivity_df.columns:
            axes[1, 1].plot(sensitivity_df[parameter], sensitivity_df['total_demand'], 'purple', marker='o')
            axes[1, 1].set_xlabel(parameter)
            axes[1, 1].set_ylabel('Total Demand')
            axes[1, 1].set_title('Demand Sensitivity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
        
    def plot_pareto_frontier(
        self, 
        pareto_df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        title: str = "Pareto Frontier"
    ) -> plt.Figure:
        """Plot Pareto frontier for multi-objective optimization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all points
        ax.scatter(pareto_df[x_metric], pareto_df[y_metric], 
                  alpha=0.6, s=50, label='All Solutions')
        
        # Highlight Pareto optimal points
        if 'pareto_optimal' in pareto_df.columns:
            pareto_points = pareto_df[pareto_df['pareto_optimal'] == True]
            ax.scatter(pareto_points[x_metric], pareto_points[y_metric], 
                      color='red', s=100, label='Pareto Optimal', zorder=5)
        
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    def create_interactive_dashboard(
        self, 
        data: Dict[str, pd.DataFrame],
        title: str = "Price Optimization Dashboard"
    ) -> go.Figure:
        """Create interactive Plotly dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price-Demand Curve', 'Revenue vs Profit', 
                          'Feature Importance', 'Model Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces based on available data
        if 'price_demand' in data:
            df = data['price_demand']
            fig.add_trace(
                go.Scatter(x=df['price'], y=df['demand'], 
                          mode='lines', name='Demand Curve'),
                row=1, col=1
            )
        
        if 'revenue_profit' in data:
            df = data['revenue_profit']
            fig.add_trace(
                go.Scatter(x=df['price'], y=df['revenue'], 
                          mode='lines', name='Revenue'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['price'], y=df['profit'], 
                          mode='lines', name='Profit'),
                row=1, col=2, secondary_y=True
            )
        
        if 'feature_importance' in data:
            df = data['feature_importance']
            fig.add_trace(
                go.Bar(x=df['importance'], y=df['feature'], 
                      orientation='h', name='Feature Importance'),
                row=2, col=1
            )
        
        if 'model_performance' in data:
            df = data['model_performance']
            fig.add_trace(
                go.Scatter(x=df['actual'], y=df['predicted'], 
                          mode='markers', name='Actual vs Predicted'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800
        )
        
        return fig
        
    def save_plots(self, figures: List[plt.Figure], output_dir: str) -> None:
        """Save matplotlib figures to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, fig in enumerate(figures):
            filename = f"plot_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")


class BusinessDashboard:
    """Business-focused dashboard for price optimization."""
    
    def __init__(self):
        """Initialize business dashboard."""
        self.metrics: Dict[str, float] = {}
        
    def add_kpi(self, name: str, value: float, target: Optional[float] = None) -> None:
        """Add KPI to dashboard."""
        self.metrics[name] = {
            'value': value,
            'target': target,
            'achievement': (value / target * 100) if target else None
        }
        
    def create_kpi_dashboard(self) -> go.Figure:
        """Create KPI dashboard."""
        if not self.metrics:
            return go.Figure()
            
        # Create subplots for KPIs
        n_metrics = len(self.metrics)
        cols = min(4, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(self.metrics.keys()),
            specs=[[{"type": "indicator"}] * cols] * rows
        )
        
        for i, (name, data) in enumerate(self.metrics.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=data['value'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': name},
                    gauge={
                        'axis': {'range': [None, data['target'] * 1.2] if data['target'] else None},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, data['target']], 'color': "lightgray"} if data['target'] else None
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': data['target']
                        } if data['target'] else None
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Price Optimization KPIs",
            height=200 * rows
        )
        
        return fig
        
    def create_revenue_waterfall(self, revenue_components: Dict[str, float]) -> go.Figure:
        """Create revenue waterfall chart."""
        fig = go.Figure(go.Waterfall(
            name="Revenue",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Base Revenue", "Price Optimization", "Volume Impact", "Total"],
            y=[revenue_components.get('base', 0), 
               revenue_components.get('price_impact', 0),
               revenue_components.get('volume_impact', 0),
               revenue_components.get('total', 0)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Revenue Waterfall",
            showlegend=False,
            height=400
        )
        
        return fig
