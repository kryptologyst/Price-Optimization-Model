"""Streamlit demo application for price optimization model."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import PriceOptimizationDataGenerator, PriceOptimizationConfig, create_sample_data
from src.features import PriceOptimizationFeatureEngineer
from src.models import (
    ElasticityModel, RandomForestDemandModel, XGBoostDemandModel,
    LightGBMDemandModel, GradientBoostingDemandModel, ModelEnsemble,
    PriceOptimizationModel, create_model_ensemble
)
from src.optimization import (
    PriceOptimizationWithConstraints, MultiObjectiveOptimizer, SensitivityAnalyzer
)
from src.eval import PriceOptimizationEvaluator, BusinessMetricsCalculator
from src.viz import PriceOptimizationVisualizer, BusinessDashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Price Optimization Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">Price Optimization Model</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This is a research and educational tool for price optimization analysis. 
        It is NOT intended for automated decision-making without human review. All recommendations should be 
        validated by business experts before implementation. This model is experimental and may not reflect 
        real-world market conditions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.header("Data Parameters")
    n_products = st.sidebar.slider("Number of Products", 10, 100, 20)
    n_customers = st.sidebar.slider("Number of Customers", 100, 2000, 500)
    n_periods = st.sidebar.slider("Number of Time Periods", 30, 365, 90)
    random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    objective = st.sidebar.selectbox("Optimization Objective", ["revenue", "profit"])
    include_constraints = st.sidebar.checkbox("Include Business Constraints", value=True)
    
    if include_constraints:
        min_margin = st.sidebar.slider("Minimum Margin (%)", 0.0, 50.0, 20.0)
        min_revenue = st.sidebar.number_input("Minimum Revenue", value=10000, min_value=0)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", "Model Training", "Optimization", "Evaluation", "Visualizations"
    ])
    
    with tab1:
        show_data_overview(n_products, n_customers, n_periods, random_seed)
    
    with tab2:
        show_model_training()
    
    with tab3:
        show_optimization(objective, include_constraints, min_margin if include_constraints else None, 
                         min_revenue if include_constraints else None)
    
    with tab4:
        show_evaluation()
    
    with tab5:
        show_visualizations()

@st.cache_data
def generate_sample_data(n_products, n_customers, n_periods, random_seed):
    """Generate sample data with caching."""
    config = PriceOptimizationConfig(
        n_products=n_products,
        n_customers=n_customers,
        n_periods=n_periods,
        random_seed=random_seed
    )
    
    generator = PriceOptimizationDataGenerator(config)
    return generator.generate_all_data()

def show_data_overview(n_products, n_customers, n_periods, random_seed):
    """Show data overview tab."""
    st.header("Data Overview")
    
    with st.spinner("Generating sample data..."):
        data = generate_sample_data(n_products, n_customers, n_periods, random_seed)
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products", len(data["catalog"]))
    with col2:
        st.metric("Customers", len(data["segments"]) * 100)  # Approximate
    with col3:
        st.metric("Transactions", len(data["transactions"]))
    with col4:
        st.metric("Time Periods", n_periods)
    
    # Data preview
    st.subheader("Product Catalog")
    st.dataframe(data["catalog"].head(10))
    
    st.subheader("Customer Segments")
    st.dataframe(data["segments"])
    
    st.subheader("Transaction Sample")
    st.dataframe(data["transactions"].head(10))
    
    # Store data in session state
    st.session_state.data = data

def show_model_training():
    """Show model training tab."""
    st.header("Model Training")
    
    if "data" not in st.session_state:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    data = st.session_state.data
    
    # Feature engineering
    with st.spinner("Engineering features..."):
        feature_engineer = PriceOptimizationFeatureEngineer()
        features_df = feature_engineer.engineer_features(
            data["transactions"], 
            data["catalog"], 
            data["segments"],
            include_lags=False  # Simplified for demo
        )
    
    # Prepare training data
    X, y, feature_names = feature_engineer.prepare_model_features(features_df)
    
    # Train models
    st.subheader("Training Models")
    
    models = {
        "Elasticity Model": ElasticityModel(),
        "Random Forest": RandomForestDemandModel(),
        "XGBoost": XGBoostDemandModel(),
        "LightGBM": LightGBMDemandModel(),
        "Gradient Boosting": GradientBoostingDemandModel()
    }
    
    model_results = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(X, y, feature_names)
            predictions = model.predict(X)
            
            # Calculate metrics
            mae = np.mean(np.abs(y - predictions))
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            model_results[name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "model": model
            }
    
    # Display results
    results_df = pd.DataFrame({
        name: {metric: results[metric] for metric in ["mae", "rmse", "r2"]}
        for name, results in model_results.items()
    }).T
    
    st.subheader("Model Performance")
    st.dataframe(results_df.round(4))
    
    # Best model
    best_model_name = results_df["r2"].idxmax()
    best_model = model_results[best_model_name]["model"]
    
    st.success(f"Best performing model: {best_model_name} (R² = {results_df.loc[best_model_name, 'r2']:.4f})")
    
    # Store best model
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    st.session_state.feature_names = feature_names

def show_optimization(objective, include_constraints, min_margin, min_revenue):
    """Show optimization tab."""
    st.header("Price Optimization")
    
    if "best_model" not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    best_model = st.session_state.best_model
    data = st.session_state.data
    
    # Create optimizer
    optimizer = PriceOptimizationModel(best_model)
    optimizer.set_cost_data(data["catalog"])
    
    # Add constraints if requested
    if include_constraints:
        constraint_optimizer = PriceOptimizationWithConstraints(best_model, data["catalog"])
        
        if min_margin:
            constraint_optimizer.add_margin_constraint(min_margin / 100)
        if min_revenue:
            constraint_optimizer.add_revenue_constraint(min_revenue)
        
        # Optimize with constraints
        with st.spinner("Optimizing prices with constraints..."):
            result = constraint_optimizer.optimize_prices(objective)
    else:
        # Simple optimization without constraints
        with st.spinner("Optimizing prices..."):
            # Sample a few products for demo
            sample_products = data["catalog"].head(5)
            product_features = {}
            
            for _, product in sample_products.iterrows():
                # Create dummy features for optimization
                features = np.random.rand(10)  # Simplified
                product_features[product["product_id"]] = features
            
            result = optimizer.batch_optimize(product_features, objective=objective)
    
    # Display results
    if include_constraints and result.get("success", False):
        st.subheader("Optimization Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"${result.get('total_revenue', 0):,.2f}")
        with col2:
            st.metric("Total Profit", f"${result.get('total_profit', 0):,.2f}")
        with col3:
            st.metric("Total Demand", f"{result.get('total_demand', 0):,.0f}")
        with col4:
            st.metric("Average Margin", f"{result.get('average_margin', 0)*100:.1f}%")
        
        # Price recommendations
        if "optimal_values" in result:
            st.subheader("Price Recommendations")
            optimal_prices = result["optimal_values"]
            
            price_df = pd.DataFrame({
                "Product ID": data["catalog"]["product_id"].head(len(optimal_prices)),
                "Current Price": data["catalog"]["base_price"].head(len(optimal_prices)),
                "Optimal Price": optimal_prices,
                "Price Change": ((optimal_prices - data["catalog"]["base_price"].head(len(optimal_prices)).values) / 
                               data["catalog"]["base_price"].head(len(optimal_prices)).values * 100)
            })
            
            st.dataframe(price_df.round(2))
    
    else:
        st.subheader("Optimization Results")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result.round(2))
        else:
            st.error("Optimization failed. Please check constraints.")

def show_evaluation():
    """Show evaluation tab."""
    st.header("Model Evaluation")
    
    if "best_model" not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    # Create evaluator
    evaluator = PriceOptimizationEvaluator()
    
    # Business metrics calculator
    metrics_calc = BusinessMetricsCalculator()
    
    st.subheader("Business Metrics")
    
    # Sample metrics calculation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Price Elasticity", "-1.2", "Moderately Elastic")
    
    with col2:
        st.metric("Market Penetration", "15.3%", "+2.1%")
    
    with col3:
        st.metric("Customer Lifetime Value", "$1,250", "+$150")
    
    # A/B Test Simulation
    st.subheader("A/B Test Simulation")
    
    # Generate sample A/B test data
    np.random.seed(42)
    control_group = np.random.normal(100, 15, 1000)
    treatment_group = np.random.normal(105, 15, 1000)
    
    # Run statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Control Mean", f"${np.mean(control_group):.2f}")
    with col2:
        st.metric("Treatment Mean", f"${np.mean(treatment_group):.2f}")
    with col3:
        st.metric("P-value", f"{p_value:.4f}", "Significant" if p_value < 0.05 else "Not Significant")
    
    # Lift calculation
    lift = ((np.mean(treatment_group) - np.mean(control_group)) / np.mean(control_group)) * 100
    st.metric("Revenue Lift", f"{lift:.1f}%")

def show_visualizations():
    """Show visualizations tab."""
    st.header("Visualizations")
    
    if "data" not in st.session_state:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    data = st.session_state.data
    
    # Create visualizer
    visualizer = PriceOptimizationVisualizer()
    
    # Price-demand curve
    st.subheader("Price-Demand Analysis")
    
    # Sample product for demonstration
    sample_product = data["catalog"].iloc[0]
    prices = np.linspace(sample_product["base_price"] * 0.5, sample_product["base_price"] * 2, 50)
    
    # Calculate demand using elasticity
    base_demand = 100
    demands = base_demand * (prices / sample_product["base_price"]) ** sample_product["elasticity"]
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=demands, mode='lines', name='Demand Curve'))
    
    # Find optimal point
    revenues = prices * demands
    optimal_idx = np.argmax(revenues)
    optimal_price = prices[optimal_idx]
    optimal_demand = demands[optimal_idx]
    
    fig.add_trace(go.Scatter(x=[optimal_price], y=[optimal_demand], 
                            mode='markers', name='Optimal Point', 
                            marker=dict(size=15, color='red')))
    
    fig.update_layout(
        title=f"Price-Demand Curve for {sample_product['name']}",
        xaxis_title="Price ($)",
        yaxis_title="Demand",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue and profit curves
    st.subheader("Revenue vs Profit Analysis")
    
    profits = revenues - (sample_product["cost"] * demands)
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=prices, y=revenues, mode='lines', name='Revenue'),
        secondary_y=False
    )
    
    fig2.add_trace(
        go.Scatter(x=prices, y=profits, mode='lines', name='Profit'),
        secondary_y=True
    )
    
    fig2.update_xaxes(title_text="Price ($)")
    fig2.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig2.update_yaxes(title_text="Profit ($)", secondary_y=True)
    fig2.update_layout(title_text="Revenue and Profit Curves")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Product performance comparison
    st.subheader("Product Performance Comparison")
    
    # Calculate metrics for all products
    product_metrics = []
    for _, product in data["catalog"].iterrows():
        # Sample transactions for this product
        product_transactions = data["transactions"][data["transactions"]["product_id"] == product["product_id"]]
        
        if len(product_transactions) > 0:
            avg_price = product_transactions["price"].mean()
            total_quantity = product_transactions["quantity"].sum()
            total_revenue = (product_transactions["price"] * product_transactions["quantity"]).sum()
            
            product_metrics.append({
                "Product": product["name"],
                "Category": product["category"],
                "Avg Price": avg_price,
                "Total Quantity": total_quantity,
                "Total Revenue": total_revenue,
                "Elasticity": product["elasticity"]
            })
    
    if product_metrics:
        metrics_df = pd.DataFrame(product_metrics)
        
        # Revenue by category
        fig3 = px.bar(metrics_df.groupby("Category")["Total Revenue"].sum().reset_index(),
                     x="Category", y="Total Revenue",
                     title="Total Revenue by Category")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Price vs Revenue scatter
        fig4 = px.scatter(metrics_df, x="Avg Price", y="Total Revenue", 
                         color="Category", size="Total Quantity",
                         title="Price vs Revenue by Category",
                         hover_data=["Product", "Elasticity"])
        st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()
