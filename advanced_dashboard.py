# Advanced Interactive Dashboard for Wave Energy Farm Optimization
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class AdvancedVisualizationDashboard:
    """Advanced visualization and monitoring dashboard"""
    
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="🌊 Advanced Wave Energy Dashboard",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-good { color: #00ff00; }
        .status-warning { color: #ffa500; }
        .status-critical { color: #ff0000; }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_main_dashboard(self):
        """Render the main dashboard"""
        st.title("🌊 Advanced Wave Energy Farm Control Center")
        
        # Real-time status bar
        self.render_status_bar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎛️ Real-Time Control", 
            "📊 Performance Analytics", 
            "🗺️ 3D Farm Visualization",
            "🔧 Maintenance Hub",
            "💰 Economic Dashboard"
        ])
        
        with tab1:
            self.render_realtime_control()
        
        with tab2:
            self.render_performance_analytics()
        
        with tab3:
            self.render_3d_visualization()
        
        with tab4:
            self.render_maintenance_hub()
        
        with tab5:
            self.render_economic_dashboard()
    
    def render_status_bar(self):
        """Render real-time status indicators"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Simulate real-time data
        total_power = np.random.uniform(45, 55)  # MW
        efficiency = np.random.uniform(82, 95)  # %
        operational_wecs = np.random.randint(46, 50)
        wave_height = np.random.uniform(2.1, 3.2)  # meters
        wind_speed = np.random.uniform(8, 15)  # m/s
        grid_status = np.random.choice(["Connected", "Maintenance", "Critical"])
        
        with col1:
            st.metric("🔋 Total Power", f"{total_power:.1f} MW", 
                     f"{np.random.uniform(-2, 3):.1f}%")
        
        with col2:
            st.metric("⚡ Efficiency", f"{efficiency:.1f}%", 
                     f"{np.random.uniform(-1, 2):.1f}%")
        
        with col3:
            st.metric("🏭 Operational WECs", f"{operational_wecs}/49", 
                     f"{np.random.randint(-2, 1)}")
        
        with col4:
            st.metric("🌊 Wave Height", f"{wave_height:.1f} m", 
                     f"{np.random.uniform(-0.3, 0.5):.1f}")
        
        with col5:
            st.metric("💨 Wind Speed", f"{wind_speed:.1f} m/s", 
                     f"{np.random.uniform(-2, 3):.1f}")
        
        with col6:
            status_color = "🟢" if grid_status == "Connected" else "🟡" if grid_status == "Maintenance" else "🔴"
            st.metric("🔌 Grid Status", f"{status_color} {grid_status}")
    
    def render_realtime_control(self):
        """Render real-time control panel"""
        st.header("🎛️ Real-Time Farm Control")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Real-time power output chart
            st.subheader("📈 Live Power Output")
            
            # Generate time series data
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                end=datetime.now(), freq='10min')
            power_data = 45 + 10 * np.sin(np.arange(len(times)) * 0.1) + np.random.normal(0, 2, len(times))
            
            df_power = pd.DataFrame({
                'Time': times,
                'Power (MW)': power_data,
                'Target (MW)': [50] * len(times),
                'Minimum (MW)': [40] * len(times)
            })
            
            fig_power = go.Figure()
            
            fig_power.add_trace(go.Scatter(
                x=df_power['Time'], 
                y=df_power['Power (MW)'],
                mode='lines',
                name='Actual Power',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig_power.add_trace(go.Scatter(
                x=df_power['Time'], 
                y=df_power['Target (MW)'],
                mode='lines',
                name='Target',
                line=dict(color='green', dash='dash', width=2)
            ))
            
            fig_power.add_trace(go.Scatter(
                x=df_power['Time'], 
                y=df_power['Minimum (MW)'],
                mode='lines',
                name='Minimum Threshold',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig_power.update_layout(
                title="24-Hour Power Output Trend",
                xaxis_title="Time",
                yaxis_title="Power Output (MW)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_power, use_container_width=True)
        
        with col2:
            # Control panel
            st.subheader("🎮 Control Panel")
            
            # Emergency controls
            st.markdown("### 🚨 Emergency Controls")
            if st.button("🛑 Emergency Stop", type="primary"):
                st.warning("Emergency stop activated!")
            
            if st.button("⚡ Quick Restart"):
                st.success("System restart initiated...")
            
            # Operational controls
            st.markdown("### ⚙️ Operational Settings")
            
            power_limit = st.slider("Power Limit (%)", 50, 100, 85, 5)
            maintenance_mode = st.checkbox("Maintenance Mode")
            auto_optimization = st.checkbox("Auto Optimization", value=True)
            
            # Wave condition alerts
            st.markdown("### 🌊 Condition Alerts")
            
            alerts = [
                ("⚠️ High wave activity detected", "warning"),
                ("✅ Optimal conditions for next 6 hours", "success"),
                ("🔧 WEC-23 efficiency below threshold", "warning")
            ]
            
            for alert, alert_type in alerts:
                if alert_type == "warning":
                    st.warning(alert)
                elif alert_type == "success":
                    st.success(alert)
                else:
                    st.info(alert)
        
        # Individual WEC status grid
        st.subheader("🏭 Individual WEC Status")
        
        # Create WEC status grid
        wec_data = []
        for i in range(49):
            wec_data.append({
                'WEC_ID': f'WEC-{i+1:02d}',
                'Power (kW)': np.random.uniform(800, 1200),
                'Efficiency (%)': np.random.uniform(75, 98),
                'Status': np.random.choice(['Operational', 'Maintenance', 'Offline'], p=[0.8, 0.15, 0.05]),
                'Temperature (°C)': np.random.uniform(15, 25),
                'Vibration': np.random.choice(['Normal', 'Elevated', 'High'], p=[0.7, 0.25, 0.05])
            })
        
        df_wecs = pd.DataFrame(wec_data)
        
        # Color-code status
        def color_status(val):
            if val == 'Operational':
                return 'background-color: lightgreen'
            elif val == 'Maintenance':
                return 'background-color: yellow'
            else:
                return 'background-color: lightcoral'
        
        styled_df = df_wecs.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, height=300)
    
    def render_performance_analytics(self):
        """Render performance analytics dashboard"""
        st.header("📊 Performance Analytics & Insights")
        
        # Time period selector
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            time_period = st.selectbox("📅 Time Period", 
                                     ["Last 24 Hours", "Last Week", "Last Month", "Last Year"])
        
        with col2:
            metric_type = st.selectbox("📈 Metric Type", 
                                     ["Power Output", "Efficiency", "Revenue", "Availability"])
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        # Generate synthetic performance data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        seasonal_trend = 45 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        noise = np.random.normal(0, 3, len(dates))
        performance_data = seasonal_trend + noise
        
        df_performance = pd.DataFrame({
            'Date': dates,
            'Power_MW': performance_data,
            'Efficiency': np.random.uniform(80, 95, len(dates)),
            'Revenue_k': performance_data * 0.12 * 24,  # Assuming $0.12/kWh
            'Availability': np.random.uniform(85, 99, len(dates))
        })
        
        # Create subplot for multiple metrics
        fig_perf = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Output (MW)', 'Efficiency (%)', 'Daily Revenue ($k)', 'Availability (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Power output
        fig_perf.add_trace(
            go.Scatter(x=df_performance['Date'], y=df_performance['Power_MW'], 
                      name='Power', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Efficiency
        fig_perf.add_trace(
            go.Scatter(x=df_performance['Date'], y=df_performance['Efficiency'], 
                      name='Efficiency', line=dict(color='green')),
            row=1, col=2
        )
        
        # Revenue
        fig_perf.add_trace(
            go.Scatter(x=df_performance['Date'], y=df_performance['Revenue_k'], 
                      name='Revenue', line=dict(color='gold')),
            row=2, col=1
        )
        
        # Availability
        fig_perf.add_trace(
            go.Scatter(x=df_performance['Date'], y=df_performance['Availability'], 
                      name='Availability', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig_perf.update_layout(height=600, showlegend=False, title_text="Annual Performance Overview")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Performance Benchmarks")
            
            benchmark_data = {
                'Metric': ['Power Output', 'Efficiency', 'Availability', 'Revenue'],
                'Current': [48.2, 87.5, 94.2, 2.1],
                'Target': [50.0, 90.0, 95.0, 2.2],
                'Industry Avg': [42.0, 82.0, 88.0, 1.8],
                'Unit': ['MW', '%', '%', 'M$/month']
            }
            
            df_benchmark = pd.DataFrame(benchmark_data)
            
            fig_bench = go.Figure()
            
            fig_bench.add_trace(go.Bar(
                name='Current',
                x=df_benchmark['Metric'],
                y=df_benchmark['Current'],
                marker_color='lightblue'
            ))
            
            fig_bench.add_trace(go.Bar(
                name='Target',
                x=df_benchmark['Metric'],
                y=df_benchmark['Target'],
                marker_color='green'
            ))
            
            fig_bench.add_trace(go.Bar(
                name='Industry Average',
                x=df_benchmark['Metric'],
                y=df_benchmark['Industry Avg'],
                marker_color='gray'
            ))
            
            fig_bench.update_layout(
                title="Performance vs Benchmarks",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_bench, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Key Performance Indicators")
            
            # KPI cards
            kpis = [
                ("Capacity Factor", "34.2%", "↑ 2.1%", "success"),
                ("MTBF (Hours)", "8,760", "↑ 156", "success"),
                ("O&M Cost", "$1.2M", "↓ $0.1M", "success"),
                ("Grid Compliance", "99.8%", "↔ 0.0%", "normal")
            ]
            
            for kpi_name, kpi_value, kpi_change, kpi_status in kpis:
                delta_color = "normal" if kpi_status == "normal" else kpi_status
                st.metric(kpi_name, kpi_value, kpi_change, delta_color=delta_color)
            
            # Efficiency heatmap by WEC
            st.subheader("🗺️ WEC Efficiency Heatmap")
            
            # Generate efficiency data for each WEC
            wec_efficiency = np.random.uniform(75, 98, 49).reshape(7, 7)
            
            fig_heatmap = px.imshow(
                wec_efficiency,
                labels=dict(x="Column", y="Row", color="Efficiency %"),
                x=[f"C{i+1}" for i in range(7)],
                y=[f"R{i+1}" for i in range(7)],
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            
            fig_heatmap.update_layout(
                title="WEC Efficiency Distribution",
                height=300
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def render_3d_visualization(self):
        """Render advanced 3D farm visualization"""
        st.header("🗺️ Advanced 3D Farm Visualization")
        
        # Visualization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_mode = st.selectbox("👁️ View Mode", 
                                   ["Real-time Status", "Power Output", "Wave Conditions", "Maintenance"])
        
        with col2:
            time_animation = st.checkbox("🎬 Time Animation")
            show_connections = st.checkbox("🔗 Show Connections", value=True)
        
        with col3:
            color_scheme = st.selectbox("🎨 Color Scheme", 
                                      ["Performance", "Status", "Temperature", "Age"])
        
        # Generate 3D farm layout
        positions = [(100 + i*150, 100 + j*150, -10) for i in range(7) for j in range(7)][:49]
        
        # Create 3D visualization
        fig_3d = go.Figure()
        
        # WEC positions with status
        wec_colors = np.random.uniform(0.7, 1.0, 49)  # Performance metric
        wec_sizes = np.random.uniform(15, 25, 49)     # Power output indicator
        
        fig_3d.add_trace(go.Scatter3d(
            x=[pos[0] for pos in positions],
            y=[pos[1] for pos in positions],
            z=[pos[2] for pos in positions],
            mode='markers+text',
            marker=dict(
                size=wec_sizes,
                color=wec_colors,
                colorscale='Viridis',
                colorbar=dict(title="Performance"),
                opacity=0.8,
                line=dict(width=2, color='darkblue')
            ),
            text=[f'WEC{i+1}' for i in range(49)],
            textposition="top center",
            name='WECs',
            hovertemplate='<b>%{text}</b><br>X: %{x}m<br>Y: %{y}m<br>Z: %{z}m<br>Performance: %{marker.color:.2f}<extra></extra>'
        ))
        
        # Add seafloor
        x_floor = np.linspace(0, 1000, 20)
        y_floor = np.linspace(0, 1000, 20)
        X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
        Z_floor = -50 + 5 * np.sin(X_floor/100) * np.cos(Y_floor/100)  # Realistic seafloor
        
        fig_3d.add_trace(go.Surface(
            x=X_floor,
            y=Y_floor,
            z=Z_floor,
            colorscale='Earth',
            opacity=0.3,
            name='Seafloor',
            showscale=False
        ))
        
        # Add wave surface
        if view_mode == "Wave Conditions":
            Z_waves = 2 * np.sin(X_floor/50 + datetime.now().timestamp()) * np.cos(Y_floor/50)
            
            fig_3d.add_trace(go.Surface(
                x=X_floor,
                y=Y_floor,
                z=Z_waves,
                colorscale='Blues',
                opacity=0.4,
                name='Wave Surface',
                showscale=False
            ))
        
        # Add cable connections
        if show_connections:
            # Simplified cable routing (star topology to substation)
            substation_pos = (500, 500, -5)
            
            for pos in positions[::3]:  # Show every 3rd connection for clarity
                fig_3d.add_trace(go.Scatter3d(
                    x=[pos[0], substation_pos[0]],
                    y=[pos[1], substation_pos[1]],
                    z=[pos[2], substation_pos[2]],
                    mode='lines',
                    line=dict(color='orange', width=3),
                    showlegend=False,
                    name='Power Cable'
                ))
        
        # Add substation
        fig_3d.add_trace(go.Scatter3d(
            x=[500],
            y=[500],
            z=[-5],
            mode='markers+text',
            marker=dict(
                size=30,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='darkred')
            ),
            text=['Substation'],
            textposition="top center",
            name='Substation'
        ))
        
        fig_3d.update_layout(
            title=f"3D Wave Energy Farm - {view_mode}",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            height=700,
            showlegend=True
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Environmental conditions overlay
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌊 Environmental Conditions")
            
            # Current conditions
            conditions = {
                'Wave Height': f"{np.random.uniform(2.0, 3.5):.1f} m",
                'Wave Period': f"{np.random.uniform(7, 10):.1f} s",
                'Wave Direction': f"{np.random.uniform(260, 290):.0f}°",
                'Current Speed': f"{np.random.uniform(0.5, 1.2):.1f} m/s",
                'Water Depth': "45-55 m",
                'Tide Level': f"{np.random.uniform(-0.5, 0.5):.1f} m"
            }
            
            for condition, value in conditions.items():
                st.text(f"{condition}: {value}")
        
        with col2:
            st.subheader("⚡ Power Distribution")
            
            # Power by zone
            zones = ['Northwest', 'Northeast', 'Central', 'Southwest', 'Southeast']
            zone_power = np.random.uniform(8, 12, 5)
            
            fig_zones = px.pie(
                values=zone_power,
                names=zones,
                title="Power Distribution by Zone"
            )
            
            st.plotly_chart(fig_zones, use_container_width=True)
    
    def render_maintenance_hub(self):
        """Render maintenance management dashboard"""
        st.header("🔧 Maintenance Management Hub")
        
        # Maintenance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔧 Active Maintenance", "3", "↓ 2")
        with col2:
            st.metric("⏰ Scheduled This Week", "7", "↑ 1")
        with col3:
            st.metric("🚨 Critical Issues", "1", "↔ 0")
        with col4:
            st.metric("💰 Maintenance Cost (MTD)", "$45k", "↓ $5k")
        
        # Maintenance tabs
        maint_tab1, maint_tab2, maint_tab3 = st.tabs([
            "📋 Current Issues", "📅 Maintenance Schedule", "📊 Maintenance Analytics"
        ])
        
        with maint_tab1:
            # Current maintenance issues
            st.subheader("🔴 Active Maintenance Issues")
            
            issues_data = [
                {
                    'WEC_ID': 'WEC-23',
                    'Issue': 'Hydraulic pressure low',
                    'Severity': 'High',
                    'Reported': '2024-12-01 14:30',
                    'Technician': 'John Smith',
                    'ETA': '4 hours',
                    'Status': 'In Progress'
                },
                {
                    'WEC_ID': 'WEC-07',
                    'Issue': 'Generator bearing replacement',
                    'Severity': 'Medium',
                    'Reported': '2024-12-01 09:15',
                    'Technician': 'Sarah Johnson',
                    'ETA': '2 days',
                    'Status': 'Scheduled'
                },
                {
                    'WEC_ID': 'WEC-41',
                    'Issue': 'Control system calibration',
                    'Severity': 'Low',
                    'Reported': '2024-11-30 16:45',
                    'Technician': 'Mike Chen',
                    'ETA': '1 day',
                    'Status': 'Pending Parts'
                }
            ]
            
            df_issues = pd.DataFrame(issues_data)
            
            # Color code by severity
            def color_severity(val):
                if val == 'High':
                    return 'background-color: #ffcccc'
                elif val == 'Medium':
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #d4edda'
            
            styled_issues = df_issues.style.applymap(color_severity, subset=['Severity'])
            st.dataframe(styled_issues, use_container_width=True)
            
            # Quick actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚨 Report New Issue"):
                    st.info("Issue reporting form would open here")
            with col2:
                if st.button("👥 Assign Technician"):
                    st.info("Technician assignment interface")
            with col3:
                if st.button("📱 Send Alert"):
                    st.success("Alert sent to maintenance team")
        
        with maint_tab2:
            # Maintenance schedule
            st.subheader("📅 Maintenance Schedule")
            
            # Calendar view of maintenance
            schedule_data = []
            for i in range(30):
                date = datetime.now() + timedelta(days=i)
                if np.random.random() < 0.2:  # 20% chance of maintenance
                    schedule_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'WEC_ID': f'WEC-{np.random.randint(1, 50):02d}',
                        'Type': np.random.choice(['Preventive', 'Corrective', 'Inspection']),
                        'Duration': f"{np.random.randint(2, 8)} hours",
                        'Crew': np.random.choice(['Team A', 'Team B', 'External'])
                    })
            
            df_schedule = pd.DataFrame(schedule_data)
            
            if not df_schedule.empty:
                # Group by date
                for date in df_schedule['Date'].unique()[:7]:  # Show next 7 days
                    day_data = df_schedule[df_schedule['Date'] == date]
                    with st.expander(f"📅 {date} ({len(day_data)} activities)"):
                        st.dataframe(day_data.drop('Date', axis=1), use_container_width=True)
            
            # Schedule optimization
            if st.button("🎯 Optimize Schedule"):
                st.success("Schedule optimized for minimal production impact")
        
        with maint_tab3:
            # Maintenance analytics
            st.subheader("📊 Maintenance Performance Analytics")
            
            # MTBF trend
            dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            mtbf_data = 8000 + 500 * np.sin(np.arange(12) * 0.5) + np.random.normal(0, 200, 12)
            
            fig_mtbf = go.Figure()
            fig_mtbf.add_trace(go.Scatter(
                x=dates,
                y=mtbf_data,
                mode='lines+markers',
                name='MTBF (Hours)',
                line=dict(color='blue', width=3)
            ))
            
            fig_mtbf.update_layout(
                title="Mean Time Between Failures Trend",
                xaxis_title="Month",
                yaxis_title="MTBF (Hours)",
                height=400
            )
            
            st.plotly_chart(fig_mtbf, use_container_width=True)
            
            # Maintenance cost breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost by type
                cost_data = {
                    'Type': ['Preventive', 'Corrective', 'Emergency', 'Inspection'],
                    'Cost': [120000, 180000, 95000, 45000]
                }
                
                fig_cost = px.bar(
                    cost_data,
                    x='Type',
                    y='Cost',
                    title="Maintenance Cost by Type (YTD)"
                )
                
                st.plotly_chart(fig_cost, use_container_width=True)
            
            with col2:
                # Failure modes
                failure_modes = {
                    'Mode': ['Hydraulic', 'Electrical', 'Mechanical', 'Control', 'Structural'],
                    'Frequency': [25, 18, 22, 12, 8]
                }
                
                fig_failures = px.pie(
                    failure_modes,
                    values='Frequency',
                    names='Mode',
                    title="Failure Modes Distribution"
                )
                
                st.plotly_chart(fig_failures, use_container_width=True)
    
    def render_economic_dashboard(self):
        """Render economic analysis dashboard"""
        st.header("💰 Economic Performance Dashboard")
        
        # Economic overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💵 Monthly Revenue", "$2.1M", "↑ $0.15M")
        with col2:
            st.metric("💰 YTD Profit", "$18.5M", "↑ $2.1M")
        with col3:
            st.metric("📈 ROI", "12.4%", "↑ 0.8%")
        with col4:
            st.metric("⏳ Payback Period", "8.2 years", "↓ 0.3 years")
        
        # Economic analysis tabs
        econ_tab1, econ_tab2, econ_tab3 = st.tabs([
            "💰 Revenue Analysis", "💸 Cost Management", "📊 Financial Projections"
        ])
        
        with econ_tab1:
            # Revenue analysis
            st.subheader("💰 Revenue Performance")
            
            # Monthly revenue trend
            months = pd.date_range(start='2024-01-01', periods=12, freq='M')
            revenue_data = 1.8 + 0.4 * np.sin(np.arange(12) * 0.5) + np.random.normal(0, 0.1, 12)
            
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Scatter(
                x=months,
                y=revenue_data,
                mode='lines+markers',
                name='Monthly Revenue',
                line=dict(color='green', width=3),
                fill='tonexty'
            ))
            
            fig_revenue.update_layout(
                title="Monthly Revenue Trend (2024)",
                xaxis_title="Month",
                yaxis_title="Revenue (Million $)",
                height=400
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
            
            # Revenue sources
            col1, col2 = st.columns(2)
            
            with col1:
                revenue_sources = {
                    'Source': ['Energy Sales', 'Grid Services', 'RECs', 'Subsidies'],
                    'Amount': [1.65, 0.25, 0.15, 0.05]
                }
                
                fig_sources = px.pie(
                    revenue_sources,
                    values='Amount',
                    names='Source',
                    title="Revenue Sources (Monthly)"
                )
                
                st.plotly_chart(fig_sources, use_container_width=True)
            
            with col2:
                # Price optimization
                st.subheader("⚡ Energy Price Analysis")
                
                hours = range(24)
                prices = 0.12 + 0.04 * np.sin(np.array(hours) * np.pi / 12) + np.random.normal(0, 0.01, 24)
                production = 45 + 10 * np.sin(np.array(hours) * np.pi / 8) + np.random.normal(0, 2, 24)
                
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=hours,
                    y=prices,
                    mode='lines',
                    name='Price ($/kWh)',
                    yaxis='y'
                ))
                
                fig_price.add_trace(go.Scatter(
                    x=hours,
                    y=production,
                    mode='lines',
                    name='Production (MW)',
                    yaxis='y2'
                ))
                
                fig_price.update_layout(
                    title="Hourly Price vs Production",
                    xaxis_title="Hour of Day",
                    yaxis=dict(title="Price ($/kWh)", side="left"),
                    yaxis2=dict(title="Production (MW)", side="right", overlaying="y"),
                    height=400
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
        
        with econ_tab2:
            # Cost management
            st.subheader("💸 Cost Analysis & Management")
            
            # Cost breakdown
            cost_categories = {
                'Category': ['O&M', 'Insurance', 'Grid Fees', 'Financing', 'Admin'],
                'Monthly_Cost': [450000, 85000, 125000, 310000, 65000],
                'Budget': [480000, 80000, 120000, 320000, 70000]
            }
            
            df_costs = pd.DataFrame(cost_categories)
            
            fig_costs = go.Figure()
            fig_costs.add_trace(go.Bar(
                name='Actual',
                x=df_costs['Category'],
                y=df_costs['Monthly_Cost'],
                marker_color='lightcoral'
            ))
            
            fig_costs.add_trace(go.Bar(
                name='Budget',
                x=df_costs['Category'],
                y=df_costs['Budget'],
                marker_color='lightblue'
            ))
            
            fig_costs.update_layout(
                title="Monthly Costs vs Budget",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_costs, use_container_width=True)
            
            # Cost optimization suggestions
            st.subheader("🎯 Cost Optimization Opportunities")
            
            opportunities = [
                ("🔧 Predictive Maintenance", "Potential savings: $125k/year", "Reduce unplanned maintenance by 30%"),
                ("⚡ Grid Optimization", "Potential savings: $45k/year", "Optimize grid connection fees"),
                ("🤝 Insurance Review", "Potential savings: $28k/year", "Renegotiate coverage terms"),
                ("🔄 Spare Parts Inventory", "Potential savings: $65k/year", "Optimize inventory levels")
            ]
            
            for title, savings, description in opportunities:
                with st.expander(f"{title} - {savings}"):
                    st.write(description)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(np.random.uniform(0.3, 0.8))
                    with col2:
                        if st.button("Implement", key=f"impl_{title}"):
                            st.success("Implementation scheduled")
        
        with econ_tab3:
            # Financial projections
            st.subheader("📊 Financial Projections & Scenarios")
            
            # Scenario analysis
            scenarios = st.multiselect(
                "Select Scenarios to Compare",
                ["Base Case", "Optimistic", "Pessimistic", "High Maintenance", "Technology Upgrade"],
                default=["Base Case", "Optimistic", "Pessimistic"]
            )
            
            # Generate projection data
            years = list(range(2024, 2035))
            projections = {}
            
            for scenario in scenarios:
                if scenario == "Base Case":
                    growth_rate = 0.02
                    base_revenue = 25
                elif scenario == "Optimistic":
                    growth_rate = 0.05
                    base_revenue = 28
                elif scenario == "Pessimistic":
                    growth_rate = -0.01
                    base_revenue = 22
                elif scenario == "High Maintenance":
                    growth_rate = 0.01
                    base_revenue = 23
                else:  # Technology Upgrade
                    growth_rate = 0.08
                    base_revenue = 32
                
                revenue_projection = [base_revenue * (1 + growth_rate) ** i for i in range(len(years))]
                projections[scenario] = revenue_projection
            
            # Plot projections
            fig_proj = go.Figure()
            
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            for i, (scenario, values) in enumerate(projections.items()):
                fig_proj.add_trace(go.Scatter(
                    x=years,
                    y=values,
                    mode='lines+markers',
                    name=scenario,
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            fig_proj.update_layout(
                title="Revenue Projections by Scenario",
                xaxis_title="Year",
                yaxis_title="Annual Revenue (Million $)",
                height=500
            )
            
            st.plotly_chart(fig_proj, use_container_width=True)
            
            # NPV analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💎 NPV Analysis")
                
                discount_rate = st.slider("Discount Rate (%)", 5.0, 15.0, 8.0, 0.5) / 100
                
                npv_data = {}
                for scenario, revenues in projections.items():
                    # Simple NPV calculation
                    cash_flows = [r - 15 for r in revenues]  # Assume $15M annual costs
                    npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
                    npv_data[scenario] = npv
                
                fig_npv = px.bar(
                    x=list(npv_data.keys()),
                    y=list(npv_data.values()),
                    title=f"NPV Comparison (@ {discount_rate:.1%} discount rate)"
                )
                
                st.plotly_chart(fig_npv, use_container_width=True)
            
            with col2:
                st.subheader("📈 Sensitivity Analysis")
                
                # Key variables impact on NPV
                variables = ['Electricity Price', 'Capacity Factor', 'O&M Costs', 'Discount Rate']
                impact = [15.2, 12.8, -8.5, -11.3]  # % change in NPV for 10% change in variable
                
                fig_sensitivity = px.bar(
                    x=variables,
                    y=impact,
                    title="NPV Sensitivity (% change for 10% variable change)",
                    color=impact,
                    color_continuous_scale="RdYlGn"
                )
                
                st.plotly_chart(fig_sensitivity, use_container_width=True)

# Main application function
def main():
    dashboard = AdvancedVisualizationDashboard()
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main()
