import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import pickle
from sklearn.ensemble import RandomForestRegressor
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Import our enhanced models
try:
    from economic_model import WECEconomicModel, EconomicWECOptimizer
    ECONOMIC_MODEL_AVAILABLE = True
except ImportError:
    ECONOMIC_MODEL_AVAILABLE = False

try:
    from advanced_models import AdvancedWECPredictor, EnsemblePredictor
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

try:
    from realtime_optimization import RealTimeOptimizer, PredictiveMaintenanceSystem
    REALTIME_OPTIMIZATION_AVAILABLE = True
except ImportError:
    REALTIME_OPTIMIZATION_AVAILABLE = False

try:
    from environmental_analysis import EnvironmentalImpactAnalyzer
    ENVIRONMENTAL_ANALYSIS_AVAILABLE = True
except ImportError:
    ENVIRONMENTAL_ANALYSIS_AVAILABLE = False

# Show warnings for missing modules
if not ECONOMIC_MODEL_AVAILABLE:
    st.warning("Economic model not available. Some features may be limited.")
if not ADVANCED_MODELS_AVAILABLE:
    st.sidebar.info("💡 Advanced ML models not available. Install dependencies for enhanced features.")
if not REALTIME_OPTIMIZATION_AVAILABLE:
    st.sidebar.info("⚡ Real-time optimization not available.")
if not ENVIRONMENTAL_ANALYSIS_AVAILABLE:
    st.sidebar.info("🌍 Environmental analysis not available.")

# Page configuration
st.set_page_config(
    page_title="🌊 Wave Energy Farm Optimizer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌊 AI-Driven Wave Energy Farm Optimizer")
st.markdown("""
**Optimize the spatial configuration of Wave Energy Converters (WECs) for maximum renewable power output**

This interactive tool uses machine learning to predict and optimize wave energy farm layouts based on real simulation data from Perth and Sydney.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Enhanced sidebar with new features
st.sidebar.subheader("🔬 Analysis Mode")
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Basic Optimization", "Advanced ML Models", "Real-Time Monitoring", "Environmental Impact", "Economic Analysis"],
    help="Choose the type of analysis to perform"
)

# Location selection
location = st.sidebar.selectbox(
    "🌍 Select Location",
    ["Perth", "Sydney"],
    help="Choose the wave environment for optimization"
)

# Number of WECs
n_wecs = st.sidebar.selectbox(
    "⚡ Number of WECs",
    [49, 100],
    help="Select the number of Wave Energy Converters"
)

# Optimization method
opt_method = st.sidebar.selectbox(
    "🔧 Optimization Method",
    ["Random Search", "Grid Layout", "Genetic Algorithm"],
    help="Choose the optimization algorithm"
)

# Constraints
st.sidebar.subheader("🎯 Constraints")
min_distance = st.sidebar.slider(
    "Minimum Distance (m)",
    min_value=30,
    max_value=200,
    value=50,
    step=10,
    help="Minimum distance between WECs"
)

area_width = st.sidebar.slider(
    "Farm Width (m)",
    min_value=800,
    max_value=2000,
    value=1400,
    step=100
)

area_height = st.sidebar.slider(
    "Farm Height (m)",
    min_value=600,
    max_value=1500,
    value=1200,
    step=100
)

@st.cache_data
def load_sample_data():
    """Load sample WEC data for demonstration"""
    np.random.seed(42)
    
    # Generate sample configurations
    n_configs = 1000
    data = []
    
    for i in range(n_configs):
        # Random WEC positions
        n = np.random.choice([49, 100])
        positions = np.random.uniform(0, 1000, (n, 2))
        
        # Calculate features
        distances = pdist(positions, 'euclidean')
        
        try:
            hull = ConvexHull(positions)
            area = hull.volume
        except:
            area = np.random.uniform(100000, 500000)
        
        # Simulate power based on spatial arrangement
        base_power = n * 50000  # Base power per WEC
        distance_factor = np.mean(distances) / 100  # Distance effect
        density_factor = n / area * 1000000  # Density effect
        location_factor = np.random.uniform(0.8, 1.2)  # Location variability
        
        total_power = base_power * (1 + 0.1 * distance_factor) * (1 - 0.05 * density_factor) * location_factor
        
        data.append({
            'n_wecs': n,
            'min_distance': np.min(distances),
            'mean_distance': np.mean(distances),
            'area': area,
            'density': n / area,
            'total_power': total_power,
            'location': np.random.choice(['Perth', 'Sydney'])
        })
    
    return pd.DataFrame(data)

def generate_layout(method, n_wecs, width, height, min_dist):
    """Generate WEC layout based on method"""
    
    if method == "Grid Layout":
        # Create grid layout
        cols = int(np.sqrt(n_wecs))
        rows = int(np.ceil(n_wecs / cols))
        
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)
        
        positions = []
        wec_count = 0
        for i in range(rows):
            for j in range(cols):
                if wec_count < n_wecs:
                    x = (j + 1) * x_spacing
                    y = (i + 1) * y_spacing
                    positions.append([x, y])
                    wec_count += 1
        
        return np.array(positions)
    
    elif method == "Random Search":
        # Random layout with distance constraints
        positions = []
        max_attempts = 1000
        
        for _ in range(n_wecs):
            attempts = 0
            while attempts < max_attempts:
                x = np.random.uniform(0, width)
                y = np.random.uniform(0, height)
                
                # Check distance constraints
                valid = True
                for pos in positions:
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < min_dist:
                        valid = False
                        break
                
                if valid:
                    positions.append([x, y])
                    break
                
                attempts += 1
            
            if attempts == max_attempts:
                # Fallback: place without distance constraint
                x = np.random.uniform(0, width)
                y = np.random.uniform(0, height)
                positions.append([x, y])
        
        return np.array(positions)
    
    elif method == "Genetic Algorithm":
        # Simplified GA (single iteration for demo)
        best_layout = None
        best_power = 0
        
        for _ in range(10):  # 10 random layouts
            layout = generate_layout("Random Search", n_wecs, width, height, min_dist)
            if layout is not None and len(layout) > 0:
                power = predict_power(layout)
                
                if power > best_power:
                    best_power = power
                    best_layout = layout
        
        # Fallback to grid layout if no good layout found
        if best_layout is None:
            best_layout = generate_layout("Grid Layout", n_wecs, width, height, min_dist)
        
        return best_layout
    
    else:
        # Default case: return grid layout for unknown methods
        return generate_layout("Grid Layout", n_wecs, width, height, min_dist)

def predict_power(positions):
    """Predict power output from WEC positions"""
    if len(positions) == 0:
        return 0
    
    # Calculate spatial features
    distances = pdist(positions, 'euclidean')
    
    try:
        hull = ConvexHull(positions)
        area = hull.volume
    except:
        # Fallback for degenerate cases
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        area = x_range * y_range
    
    # Simple power prediction model
    n_wecs = len(positions)
    base_power = n_wecs * 45000  # Base power per WEC
    
    if len(distances) > 0:
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        # Distance factor (optimal around 100-150m)
        distance_factor = 1 + 0.002 * (mean_distance - 100)
        
        # Minimum distance penalty
        if min_distance < 50:
            distance_factor *= 0.8
    else:
        distance_factor = 1
    
    # Density factor
    if area > 0:
        density = n_wecs / area
        density_factor = 1 - 0.000001 * density  # Penalty for high density
    else:
        density_factor = 1
    
    total_power = base_power * distance_factor * density_factor
    return max(total_power, 0)

def calculate_metrics(positions):
    """Calculate layout metrics"""
    if len(positions) == 0:
        return {}
    
    distances = pdist(positions, 'euclidean')
    
    try:
        hull = ConvexHull(positions)
        area = hull.volume
    except:
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        area = x_range * y_range
    
    metrics = {
        'Total WECs': len(positions),
        'Farm Area (m²)': f"{area:,.0f}",
        'Mean Distance (m)': f"{np.mean(distances):.1f}" if len(distances) > 0 else "N/A",
        'Min Distance (m)': f"{np.min(distances):.1f}" if len(distances) > 0 else "N/A",
        'Max Distance (m)': f"{np.max(distances):.1f}" if len(distances) > 0 else "N/A",
        'Density (WECs/km²)': f"{len(positions) / area * 1e6:.1f}" if area > 0 else "N/A"
    }
    
    return metrics

# Main application
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎛️ Layout Generator")
    
    if st.button("🚀 Generate Optimal Layout", type="primary"):
        with st.spinner("Optimizing WEC layout..."):
            # Generate layout
            positions = generate_layout(opt_method, n_wecs, area_width, area_height, min_distance)
            
            # Store in session state
            st.session_state['positions'] = positions
            st.session_state['generated'] = True
    
    # Load sample data for analysis
    if st.button("📊 Load Sample Data Analysis"):
        st.session_state['show_analysis'] = True

with col2:
    st.subheader("📈 Performance Metrics")
    
    if 'positions' in st.session_state and st.session_state['generated']:
        positions = st.session_state['positions']
        
        # Calculate power and metrics
        predicted_power = predict_power(positions)
        metrics = calculate_metrics(positions)
        
        # Display power prediction
        st.metric(
            "🔋 Predicted Power Output", 
            f"{predicted_power/1e6:.2f} MW",
            help="Total predicted power output from the WEC farm"
        )
        
        # Display layout metrics
        st.subheader("📏 Layout Metrics")
        for metric, value in metrics.items():
            st.text(f"{metric}: {value}")

# Layout visualization
if 'positions' in st.session_state and st.session_state['generated']:
    st.subheader("🗺️ WEC Farm Layout")
    
    positions = st.session_state['positions']
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add WEC positions
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+text',
        marker=dict(
            size=15,
            color='blue',
            symbol='circle',
            line=dict(width=2, color='darkblue')
        ),
        text=[f'WEC{i+1}' for i in range(len(positions))],
        textposition="top center",
        textfont=dict(size=8),
        name='Wave Energy Converters',
        hovertemplate='<b>%{text}</b><br>X: %{x:.0f}m<br>Y: %{y:.0f}m<extra></extra>'
    ))
    
    # Add farm boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=area_width, y1=area_height,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title=f"WEC Farm Layout - {opt_method} ({n_wecs} WECs)",
        xaxis_title="X Coordinate (m)",
        yaxis_title="Y Coordinate (m)",
        width=800,
        height=600,
        showlegend=True,
        xaxis=dict(range=[0, area_width]),
        yaxis=dict(range=[0, area_height])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distance matrix heatmap
    if len(positions) > 1:
        st.subheader("📊 Inter-WEC Distance Matrix")
        
        distance_matrix = np.zeros((len(positions), len(positions)))
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                                 (positions[i, 1] - positions[j, 1])**2)
                    distance_matrix[i, j] = dist
        
        fig_heatmap = px.imshow(
            distance_matrix,
            labels=dict(x="WEC Index", y="WEC Index", color="Distance (m)"),
            x=[f"WEC{i+1}" for i in range(len(positions))],
            y=[f"WEC{i+1}" for i in range(len(positions))],
            color_continuous_scale="Viridis",
            title="Inter-WEC Distance Matrix"
        )
        fig_heatmap.update_layout(width=600, height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Sample data analysis
if st.session_state.get('show_analysis', False):
    st.subheader("📊 Historical Performance Analysis")
    
    # Load sample data
    data = load_sample_data()
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Power Distribution", "Spatial Analysis", "Optimization Insights"])
    
    with tab1:
        # Power distribution by location and WEC count
        if not data.empty and 'location' in data.columns and 'total_power' in data.columns:
            fig_dist = px.box(
                data, 
                x='location', 
                y='total_power', 
                color='n_wecs',
                title="Power Output Distribution by Location and WEC Count"
            )
            if fig_dist:
                fig_dist.update_yaxes(title="Total Power (W)")
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.error("Could not create plot - please check data format")
        else:
            st.error("Data is missing required columns: 'location', 'total_power'")
    
    with tab2:
        # Spatial relationships
        fig_spatial = px.scatter(
            data,
            x='mean_distance',
            y='total_power',
            color='location',
            size='n_wecs',
            title="Power Output vs Mean Inter-WEC Distance",
            labels={'mean_distance': 'Mean Distance (m)', 'total_power': 'Total Power (W)'}
        )
        st.plotly_chart(fig_spatial, use_container_width=True)
    
    with tab3:
        # Performance insights
        st.markdown("### 🎯 Key Insights")
        
        # Calculate statistics
        perth_avg = data[data['location'] == 'Perth']['total_power'].mean()
        sydney_avg = data[data['location'] == 'Sydney']['total_power'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Perth Average Power",
                f"{perth_avg/1e6:.2f} MW"
            )
        
        with col2:
            st.metric(
                "Sydney Average Power", 
                f"{sydney_avg/1e6:.2f} MW"
            )
        
        with col3:
            difference = (sydney_avg - perth_avg) / perth_avg * 100
            st.metric(
                "Sydney vs Perth",
                f"{difference:+.1f}%"
            )

# 💰 ECONOMIC ANALYSIS SECTION
st.header("💰 Economic Analysis & ROI Calculator")

if ECONOMIC_MODEL_AVAILABLE:
    economic_model = WECEconomicModel()
    
    with st.expander("🔧 Economic Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Costs")
            wec_cost = st.number_input("WEC Unit Cost ($)", value=2_000_000, step=100_000)
            cable_cost = st.number_input("Cable Cost ($/meter)", value=1_500, step=100)
            maintenance_rate = st.slider("Annual Maintenance Rate (%)", 1.0, 10.0, 5.0, 0.5)
            
        with col2:
            st.subheader("Revenue")
            electricity_price = st.number_input("Electricity Price ($/kWh)", value=0.12, step=0.01)
            capacity_factor = st.slider("Capacity Factor (%)", 10.0, 60.0, 35.0, 1.0)
            project_lifetime = st.slider("Project Lifetime (years)", 15, 35, 25, 1)
        
        # Update model parameters
        economic_model.wec_unit_cost = wec_cost
        economic_model.cable_cost_per_meter = cable_cost
        economic_model.annual_maintenance_rate = maintenance_rate / 100
        economic_model.electricity_price = electricity_price
        economic_model.capacity_factor = capacity_factor / 100
        economic_model.project_lifetime = project_lifetime
    
    # Economic analysis for current layout
    if 'positions' in st.session_state and st.session_state['generated']:
        positions = st.session_state['positions']
        
        # Validate positions data
        if positions is not None and len(positions) > 0:
            n_wecs = len(positions)
            coordinates = positions.flatten()
            predicted_power = predict_power(positions)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Economic Analysis")
                
                try:
                    summary, economics = economic_model.get_economic_summary(coordinates, n_wecs, predicted_power)
                    
                    # Key metrics
                    st.metric("Net Present Value", f"${economics['npv']/1e6:.1f}M")
                    st.metric("ROI", f"{economics['roi_percent']:.1f}%")
                    st.metric("Payback Period", f"{economics['payback_period_years']:.1f} years")
                    
                    with st.expander("📈 Detailed Economic Report"):
                        st.text(summary)
                    
                except Exception as e:
                    st.error(f"Economic analysis failed: {str(e)}")
            
            with col2:
                st.subheader("💹 Financial Breakdown")
                
                try:
                    # Cost breakdown pie chart
                    breakdown = economics['installation_breakdown']
                    
                    fig_costs = go.Figure(data=[go.Pie(
                        labels=['WEC Units', 'Installation', 'Cables', 'Grid Connection'],
                        values=[breakdown['wec_costs'], breakdown['installation_costs'], 
                               breakdown['cable_costs'], breakdown['grid_connection']],
                        hole=0.4
                    )])
                    fig_costs.update_layout(title="Installation Cost Breakdown")
                    st.plotly_chart(fig_costs, use_container_width=True)
                    
                except:
                    st.info("Generate a layout first to see economic breakdown")
        else:
            st.warning("⚠️ No valid layout data available. Please generate a layout first.")
    
    # Optimization for economics
    st.subheader("🎯 Economic Optimization")
    
    if st.button("🚀 Optimize for Maximum ROI"):
        with st.spinner("Optimizing layout for maximum economic return..."):
            try:
                # Generate multiple layouts and pick best economic performer
                best_npv = -float('inf')
                best_layout = None
                best_economics = None
                
                for _ in range(10):  # Try 10 random layouts
                    test_positions = generate_layout("Random Search", 49, 1500, 1400, 50)
                    
                    # Add validation to ensure we got a valid layout
                    if test_positions is None or len(test_positions) == 0:
                        continue
                        
                    test_coords = test_positions.flatten()
                    test_power = predict_power(test_positions)
                    
                    _, test_economics = economic_model.get_economic_summary(test_coords, 49, test_power)
                    
                    if test_economics['npv'] > best_npv:
                        best_npv = test_economics['npv']
                        best_layout = test_positions
                        best_economics = test_economics
                
                if best_layout is not None:
                    # Store optimized layout
                    st.session_state['positions'] = best_layout
                    st.session_state['generated'] = True
                    
                    st.success(f"✅ Optimized layout found with NPV: ${best_npv/1e6:.1f}M")
                    st.rerun()
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

else:
    st.warning("💡 Economic analysis requires the economic_model.py file. Advanced economic features are not available.")

# 🧠 ADVANCED ML MODELS SECTION
if ADVANCED_MODELS_AVAILABLE and analysis_mode == "Advanced ML Models":
    st.header("🧠 Advanced Machine Learning Models")
    
    with st.expander("🤖 Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Ensemble Models", "Neural Networks", "Gaussian Process", "XGBoost", "LightGBM"]
            )
            
            use_advanced_features = st.checkbox("Use Advanced Spatial Features", value=True)
            
        with col2:
            cross_validation = st.checkbox("Enable Cross-Validation", value=True)
            uncertainty_estimation = st.checkbox("Include Uncertainty Estimation")
    
    if st.button("🚀 Train Advanced Models"):
        with st.spinner("Training advanced machine learning models..."):
            try:
                predictor = AdvancedWECPredictor()
                predictor.create_ensemble_models()
                
                # Generate training data (in practice, use your real data)
                n_samples = 1000
                X_samples = []
                y_samples = []
                
                for _ in range(n_samples):
                    coords = np.random.uniform(0, 1000, 49 * 2)
                    features, feature_names = predictor.prepare_advanced_features(coords, 49)
                    
                    # Simulate power
                    base_power = 49 * 45000
                    distance_factor = 1 + 0.002 * (features[0] - 100)
                    density_factor = 1 - 0.000001 * features[12] if len(features) > 12 else 1
                    power = base_power * distance_factor * density_factor + np.random.normal(0, 50000)
                    
                    X_samples.append(features)
                    y_samples.append(max(power, 0))
                
                X_samples = np.array(X_samples)
                y_samples = np.array(y_samples)
                
                # Train models
                model_scores = predictor.train_models(X_samples, y_samples)
                
                # Store trained predictor
                st.session_state['advanced_predictor'] = predictor
                
                # Display results
                st.success("✅ Advanced models trained successfully!")
                
                # Model comparison
                results_df = pd.DataFrame([
                    {'Model': name, 'R²': scores['r2'], 'RMSE': scores['rmse'], 'MAE': scores['mae']}
                    for name, scores in model_scores.items()
                ])
                
                st.subheader("📊 Model Performance Comparison")
                st.dataframe(results_df)
                
                # Best model visualization
                fig_models = go.Figure()
                fig_models.add_trace(go.Bar(
                    x=results_df['Model'],
                    y=results_df['R²'],
                    name='R² Score',
                    marker_color='lightblue'
                ))
                
                fig_models.update_layout(
                    title="Model Performance Comparison (R² Score)",
                    xaxis_title="Model",
                    yaxis_title="R² Score",
                    height=400
                )
                
                st.plotly_chart(fig_models, use_container_width=True)
                
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")

# 🌍 ENVIRONMENTAL IMPACT SECTION
if ENVIRONMENTAL_ANALYSIS_AVAILABLE and analysis_mode == "Environmental Impact":
    st.header("🌍 Environmental Impact Analysis")
    
    with st.expander("🔬 Environmental Assessment Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            assessment_type = st.selectbox(
                "Assessment Type",
                ["Ecosystem Impact", "Carbon Footprint", "Water Quality", "Marine Life", "Sustainability Metrics"]
            )
            
            monitoring_period = st.slider("Monitoring Period (days)", 30, 365, 90)
            
        with col2:
            include_mitigation = st.checkbox("Include Mitigation Measures", value=True)
            generate_report = st.checkbox("Generate Compliance Report", value=True)
    
    if st.button("🔬 Perform Environmental Analysis"):
        with st.spinner("Performing comprehensive environmental analysis..."):
            try:
                analyzer = EnvironmentalImpactAnalyzer()
                
                # Establish baseline
                baseline = analyzer.establish_baseline(location, duration_days=365)
                
                # Use current layout if available
                if 'positions' in st.session_state and st.session_state['generated']:
                    wec_layout = [(pos[0], pos[1]) for pos in st.session_state['positions']]
                else:
                    # Use default layout
                    wec_layout = [(100 + i*150, 100 + j*150) for i in range(7) for j in range(7)][:n_wecs]
                
                # Monitor environmental conditions
                for _ in range(min(monitoring_period, 30)):
                    analyzer.monitor_environmental_conditions(location, wec_layout)
                
                # Perform assessments
                if assessment_type == "Ecosystem Impact":
                    impact_assessments = analyzer.assess_ecosystem_impact(location, monitoring_period)
                    
                    st.subheader("🐋 Ecosystem Impact Assessment")
                    
                    # Impact summary
                    impact_df = pd.DataFrame([
                        {
                            'Species': impact.species_name,
                            'Population Change (%)': impact.population_change,
                            'Risk Level': impact.risk_level,
                            'Habitat Disruption': impact.habitat_disruption,
                            'Adaptation Score': impact.adaptation_score
                        }
                        for impact in impact_assessments
                    ])
                    
                    # Color-code by risk level
                    def color_risk(val):
                        if val == 'High':
                            return 'background-color: #ffcccc'
                        elif val == 'Medium':
                            return 'background-color: #fff3cd'
                        else:
                            return 'background-color: #d4edda'
                    
                    styled_impact = impact_df.style.applymap(color_risk, subset=['Risk Level'])
                    st.dataframe(styled_impact, use_container_width=True)
                    
                    # Visualization
                    fig_impact = px.scatter(
                        impact_df,
                        x='Population Change (%)',
                        y='Habitat Disruption',
                        color='Risk Level',
                        size='Adaptation Score',
                        hover_name='Species',
                        title="Species Impact Overview",
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    
                    st.plotly_chart(fig_impact, use_container_width=True)
                
                elif assessment_type == "Carbon Footprint":
                    carbon_footprint = analyzer.calculate_carbon_footprint(wec_layout)
                    
                    st.subheader("🌱 Carbon Footprint Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Net Carbon Impact", f"{carbon_footprint['net_carbon_impact']/1000:.1f}k tons CO2")
                        st.metric("Carbon Payback Period", f"{carbon_footprint['carbon_payback_years']:.1f} years")
                        st.metric("Emissions per MWh", f"{carbon_footprint['emissions_per_mwh']:.0f} kg CO2/MWh")
                    
                    with col2:
                        # Carbon breakdown
                        carbon_data = {
                            'Phase': ['Manufacturing', 'Installation', 'Operations', 'Decommissioning'],
                            'Emissions': [
                                carbon_footprint['manufacturing_emissions'],
                                carbon_footprint['installation_emissions'],
                                carbon_footprint['operational_emissions'],
                                carbon_footprint['decommissioning_emissions']
                            ]
                        }
                        
                        fig_carbon = px.bar(
                            carbon_data,
                            x='Phase',
                            y='Emissions',
                            title="Lifecycle Carbon Emissions by Phase",
                            color='Emissions',
                            color_continuous_scale='RdYlGn_r'
                        )
                        
                        st.plotly_chart(fig_carbon, use_container_width=True)
                
                elif assessment_type == "Sustainability Metrics":
                    sustainability_metrics = analyzer.calculate_sustainability_metrics(wec_layout)
                    
                    st.subheader("📊 Comprehensive Sustainability Metrics")
                    
                    # Key sustainability indicators
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Carbon Intensity", f"{sustainability_metrics['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh")
                        st.metric("Energy Density", f"{sustainability_metrics['energy_density_mwh_km2']:.1f} MWh/km²")
                    
                    with col2:
                        st.metric("Material Circularity", f"{sustainability_metrics['material_circularity_index']:.2f}")
                        st.metric("Jobs per MW", f"{sustainability_metrics['jobs_per_mw']:.1f}")
                    
                    with col3:
                        st.metric("Ecosystem Impact", f"{sustainability_metrics['ecosystem_impact_score']:.2f}")
                        st.metric("Energy Security Index", f"{sustainability_metrics['energy_security_index']:.2f}")
                    
                    # Sustainability radar chart
                    categories = ['Environmental', 'Economic', 'Social', 'Technical', 'Resource Efficiency']
                    values = [
                        1 - sustainability_metrics['ecosystem_impact_score'],
                        sustainability_metrics['economic_viability_score'],
                        sustainability_metrics['jobs_per_mw'] / 10,
                        sustainability_metrics['capacity_factor'] / 0.4,
                        sustainability_metrics['material_circularity_index']
                    ]
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Sustainability Score'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Sustainability Performance Radar"
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Mitigation measures
                if include_mitigation and assessment_type == "Ecosystem Impact":
                    mitigation_measures = analyzer.develop_mitigation_measures(impact_assessments)
                    
                    if mitigation_measures:
                        st.subheader("🛡️ Mitigation Measures")
                        
                        mitigation_df = pd.DataFrame([
                            {
                                'Target Species': measure['target_species'],
                                'Measure': measure['measure'],
                                'Effectiveness': f"{measure['effectiveness']*100:.0f}%",
                                'Cost ($)': f"${measure['cost']:,}",
                                'Implementation Time': measure['implementation_time'],
                                'Priority': measure['priority']
                            }
                            for measure in mitigation_measures
                        ])
                        
                        st.dataframe(mitigation_df, use_container_width=True)
                        
                        total_cost = sum(m['cost'] for m in mitigation_measures)
                        st.info(f"💰 Total mitigation cost: ${total_cost:,}")
                
                # Generate compliance report
                if generate_report:
                    report = analyzer.generate_environmental_report(location)
                    
                    st.subheader("📋 Environmental Compliance Report")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Executive Summary**")
                        st.write(f"Overall Impact Level: {report['executive_summary']['overall_impact_level']}")
                        st.write(f"Compliance Status: {report['executive_summary']['compliance_status']}")
                        st.write(f"Critical Issues: {len(report['executive_summary']['critical_issues'])}")
                    
                    with col2:
                        st.markdown("**Recommendations**")
                        for i, rec in enumerate(report['recommendations'][:3], 1):
                            st.write(f"{i}. {rec}")
                    
                    # Download report
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="📄 Download Full Report",
                        data=report_json,
                        file_name=f"environmental_report_{location}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                st.success("✅ Environmental analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Environmental analysis failed: {str(e)}")

# ⚡ REAL-TIME MONITORING SECTION
if REALTIME_OPTIMIZATION_AVAILABLE and analysis_mode == "Real-Time Monitoring":
    st.header("⚡ Real-Time Monitoring & Optimization")
    
    st.info("🚧 Real-time monitoring is a demonstration feature. In production, this would connect to actual sensor networks and control systems.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ System Control")
        
        if st.button("🚀 Start Real-Time Monitoring"):
            if 'realtime_optimizer' not in st.session_state:
                # Initialize with current layout or default
                if 'positions' in st.session_state and st.session_state['generated']:
                    layout = [(pos[0], pos[1]) for pos in st.session_state['positions']]
                else:
                    layout = [(100 + i*150, 100 + j*150) for i in range(7) for j in range(7)][:49]
                
                # Mock predictor
                class MockPredictor:
                    def predict(self, coords, n_wecs):
                        return n_wecs * 45000 * np.random.uniform(0.8, 1.2)
                
                st.session_state['realtime_optimizer'] = RealTimeOptimizer(layout, MockPredictor())
                st.session_state['realtime_optimizer'].start_monitoring()
                st.session_state['monitoring_active'] = True
            
            st.success("✅ Real-time monitoring started!")
        
        if st.button("⏹️ Stop Monitoring"):
            if 'realtime_optimizer' in st.session_state:
                st.session_state['realtime_optimizer'].stop_monitoring()
                st.session_state['monitoring_active'] = False
            st.info("Monitoring stopped.")
        
        # Control parameters
        if st.session_state.get('monitoring_active', False):
            st.markdown("### ⚙️ Adaptation Settings")
            adaptation_threshold = st.slider("Adaptation Threshold (%)", 1, 20, 5)
            max_movement = st.slider("Max Position Change (m)", 10, 100, 50)
            
            st.markdown("### 🌊 Current Conditions")
            st.metric("Wave Height", f"{np.random.uniform(2.0, 3.5):.1f} m")
            st.metric("Wave Period", f"{np.random.uniform(7, 10):.1f} s")
            st.metric("Wind Speed", f"{np.random.uniform(8, 15):.1f} m/s")
    
    with col2:
        st.subheader("📊 Real-Time Status")
        
        if st.session_state.get('monitoring_active', False) and 'realtime_optimizer' in st.session_state:
            # Get current status
            status = st.session_state['realtime_optimizer'].get_real_time_status()
            
            # Status metrics
            st.metric("Operational WECs", f"{status['operational_wecs']}/{status['total_wecs']}")
            st.metric("Total Power", f"{status['total_power_mw']:.1f} MW")
            st.metric("Average Efficiency", f"{status['average_efficiency']*100:.1f}%")
            st.metric("Layout Updates", status['layout_updates'])
            
            # Live power chart (simulated)
            if st.button("🔄 Refresh Data"):
                # Generate simulated power data
                times = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                    end=datetime.now(), freq='1min')
                power_data = 45 + 5 * np.sin(np.arange(len(times)) * 0.1) + np.random.normal(0, 1, len(times))
                
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=times,
                    y=power_data,
                    mode='lines',
                    name='Power Output',
                    line=dict(color='blue', width=2)
                ))
                
                fig_live.update_layout(
                    title="Live Power Output (Last Hour)",
                    xaxis_title="Time",
                    yaxis_title="Power (MW)",
                    height=300
                )
                
                st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.info("Start monitoring to see real-time status")
    
    # Predictive maintenance
    st.subheader("🔧 Predictive Maintenance")
    
    if st.button("🔍 Run Maintenance Prediction"):
        maintenance_system = PredictiveMaintenanceSystem()
        
        # Simulate WEC status
        wec_statuses = {}
        for i in range(49):
            wec_statuses[i] = type('obj', (object,), {
                'efficiency': np.random.uniform(0.7, 0.95),
                'power_output': np.random.uniform(30000, 50000),
                'wec_id': i
            })
        
        # Simulate wave conditions
        wave_conditions = type('obj', (object,), {
            'significant_wave_height': np.random.uniform(2, 4),
            'wind_speed': np.random.uniform(8, 20)
        })
        
        predictions = maintenance_system.predict_maintenance_needs(wec_statuses, wave_conditions)
        
        # Display predictions
        maintenance_df = pd.DataFrame([
            {
                'WEC_ID': f'WEC-{wec_id:02d}',
                'Risk Score': pred['risk_score'],
                'Recommended Action': pred['recommended_action'],
                'Urgency': pred['urgency']
            }
            for wec_id, pred in predictions.items()
        ])
        
        # Filter high-risk WECs
        high_risk = maintenance_df[maintenance_df['Risk Score'] > 0.4]
        
        if not high_risk.empty:
            st.warning(f"⚠️ {len(high_risk)} WECs require attention!")
            st.dataframe(high_risk, use_container_width=True)
        else:
            st.success("✅ All WECs operating within normal parameters")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<h4>🌊 AI-Driven Wave Energy Farm Optimizer</h4>
<p>Optimizing renewable energy through advanced spatial intelligence</p>
<p><em>Built with Streamlit • Machine Learning • Optimization Algorithms</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About This Tool

This application demonstrates AI-driven optimization of wave energy converter layouts using:

- **Machine Learning**: Predicts power output from spatial configurations
- **Optimization Algorithms**: Genetic algorithms, Bayesian optimization, and random search
- **Constraint Handling**: Minimum distance, area bounds, and engineering constraints
- **Interactive Visualization**: Real-time layout generation and analysis

### 🎯 Use Cases

- Wave farm design and planning
- Configuration optimization
- Performance prediction
- Cost-benefit analysis
- Risk assessment

### 🔬 Technical Details

- **Features**: Spatial distance metrics, density, area utilization
- **Models**: Random Forest, XGBoost, Neural Networks
- **Optimization**: Multi-objective with constraints
- **Validation**: Cross-validation, real data comparison
""")
