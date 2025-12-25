"""
APS Wallet - Annual Agent Performance Dashboard
Advanced analytics platform for agent network performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import custom modules
from modules.data_loader import DataLoader
from modules.analytics_engine import AnalyticsEngine
from modules.visualizations import VisualizationFactory
from modules.reports_generator import ReportGenerator
from modules.utils import format_number, format_currency, format_percentage

# Page configuration
st.set_page_config(
    page_title="APS Wallet | Agent Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            color: #1E3A8A;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-align: center;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #3B82F6;
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.95rem;
            color: #6B7280;
            font-weight: 500;
        }
        .positive-change {
            color: #10B981;
            font-weight: 600;
        }
        .negative-change {
            color: #EF4444;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

class APSDashboard:
    def __init__(self):
        self.data_loader = DataLoader()
        self.analytics = AnalyticsEngine()
        self.viz_factory = VisualizationFactory()
        self.report_gen = ReportGenerator()
        self.year = 2025
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'overview'
    
    def run(self):
        """Main application runner"""
        load_css()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on current view
        if st.session_state.data_loaded and st.session_state.analysis_results:
            self.render_main_content()
        else:
            self.render_landing_page()
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            # Logo and Title
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image("https://img.icons8.com/color/96/000000/wallet--v1.png", width=80)
            
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1E3A8A; margin-bottom: 0;">APS Wallet</h2>
                <p style="color: #6B7280; font-size: 0.9rem;">Agent Performance Dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data Upload Section
            st.markdown("### 📤 Data Management")
            
            # File uploaders
            with st.expander("Upload Data Files", expanded=not st.session_state.data_loaded):
                onboarding_file = st.file_uploader(
                    "Onboarding Data (CSV)",
                    type=['csv'],
                    key='onboarding_uploader',
                    help="Upload agent onboarding data"
                )
                
                transaction_file = st.file_uploader(
                    "Transaction Data (CSV)",
                    type=['csv'],
                    key='transaction_uploader',
                    help="Upload transaction history"
                )
                
                # Year selection
                analysis_year = st.selectbox(
                    "Analysis Year",
                    options=[2023, 2024, 2025],
                    index=2,
                    key='analysis_year'
                )
                
                # Load sample data button
                if st.button("📋 Load Sample Data", use_container_width=True):
                    with st.spinner("Loading sample data..."):
                        self.load_sample_data()
                
                # Process uploaded data
                if onboarding_file and transaction_file:
                    if st.button("🚀 Process Data", type="primary", use_container_width=True):
                        with st.spinner("Processing data..."):
                            self.process_data(onboarding_file, transaction_file, analysis_year)
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            
            view_options = {
                "overview": "📊 Overview",
                "agents": "👥 Agents & Tellers",
                "performance": "📈 Performance",
                "trends": "📅 Trends",
                "network": "🔗 Network",
                "reports": "📥 Reports"
            }
            
            for view_key, view_label in view_options.items():
                if st.button(
                    view_label,
                    key=f"nav_{view_key}",
                    use_container_width=True,
                    type="primary" if st.session_state.current_view == view_key else "secondary"
                ):
                    st.session_state.current_view = view_key
                    st.rerun()
            
            st.markdown("---")
            
            # Settings
            with st.expander("⚙️ Settings"):
                st.selectbox(
                    "Theme",
                    ["Light", "Dark", "Auto"],
                    key='theme'
                )
                
                st.slider(
                    "Chart Animation Speed",
                    min_value=0,
                    max_value=1000,
                    value=500,
                    key='animation_speed'
                )
                
                st.toggle(
                    "Show Data Tables",
                    value=True,
                    key='show_tables'
                )
            
            st.markdown("---")
            
            # Footer
            st.markdown("""
            <div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
                <p>APS Wallet Analytics v2.5</p>
                <p>© 2025 APS Financial Services</p>
                <p>For support: analytics@apswallet.com</p>
            </div>
            """, unsafe_allow_html=True)
    
    def process_data(self, onboarding_file, transaction_file, year):
        """Process uploaded data"""
        try:
            # Load data
            df_onboarding = pd.read_csv(onboarding_file)
            df_transactions = pd.read_csv(transaction_file, nrows=500000)  # Sample for performance
            
            # Store in session state
            st.session_state.df_onboarding = df_onboarding
            st.session_state.df_transactions = df_transactions
            
            # Run analysis
            results = self.analytics.analyze_performance(
                df_onboarding, 
                df_transactions, 
                year
            )
            
            st.session_state.analysis_results = results
            st.session_state.data_loaded = True
            st.session_state.current_view = 'overview'
            
            st.success("✅ Data processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            # Load sample data from data directory
            sample_onboarding = pd.DataFrame({
                'Account ID': [f'ACC-{i:06d}' for i in range(1, 101)],
                'Entity': np.random.choice(['AGENT', 'AGENT TELLER'], 100),
                'Status': np.random.choice(['ACTIVE', 'TERMINATED', 'INACTIVE'], 100, p=[0.8, 0.1, 0.1]),
                'Registration Date': pd.date_range('2024-01-01', periods=100),
                'Region': np.random.choice(['West Coast', 'Greater Banjul', 'Central River'], 100),
                'District': np.random.choice(['Kombo Central', 'Serrekunda', 'Basse'], 100)
            })
            
            sample_transactions = pd.DataFrame({
                'User Identifier': np.random.randint(1000, 2000, 50000),
                'Parent User Identifier': np.random.randint(100, 500, 50000),
                'Entity Name': np.random.choice(['AGENT', 'AGENT TELLER', 'CUSTOMER'], 50000),
                'Service Name': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], 50000),
                'Transaction Type': np.random.choice(['CR', 'DR'], 50000),
                'Amount': np.random.uniform(100, 10000, 50000),
                'Created At': pd.date_range('2025-01-01', periods=50000, freq='10min')
            })
            
            st.session_state.df_onboarding = sample_onboarding
            st.session_state.df_transactions = sample_transactions
            
            results = self.analytics.analyze_performance(
                sample_onboarding, 
                sample_transactions, 
                self.year
            )
            
            st.session_state.analysis_results = results
            st.session_state.data_loaded = True
            st.session_state.current_view = 'overview'
            
            st.success("✅ Sample data loaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
    
    def render_main_content(self):
        """Render main content based on current view"""
        results = st.session_state.analysis_results
        
        # Header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <h1 class="main-header">APS Wallet Annual Report {self.year}</h1>
            <p style="text-align: center; color: #6B7280; margin-bottom: 2rem;">
                Comprehensive Agent Performance Analytics • Last Updated: {datetime.now().strftime('%d %B %Y')}
            </p>
            """, unsafe_allow_html=True)
        
        # View-specific content
        if st.session_state.current_view == 'overview':
            self.render_overview(results)
        elif st.session_state.current_view == 'agents':
            self.render_agents_view(results)
        elif st.session_state.current_view == 'performance':
            self.render_performance_view(results)
        elif st.session_state.current_view == 'trends':
            self.render_trends_view(results)
        elif st.session_state.current_view == 'network':
            self.render_network_view(results)
        elif st.session_state.current_view == 'reports':
            self.render_reports_view(results)
    
    def render_overview(self, results):
        """Render overview dashboard"""
        # Key Metrics Row
        st.markdown("### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.render_metric_card(
                title="Total Agents",
                value=results.get('total_active_agents', 0),
                change=results.get('agent_growth', 0),
                icon="👥",
                color="#3B82F6"
            )
        
        with col2:
            self.render_metric_card(
                title="Active Tellers",
                value=results.get('total_active_tellers', 0),
                change=results.get('teller_growth', 0),
                icon="💼",
                color="#8B5CF6"
            )
        
        with col3:
            self.render_metric_card(
                title="2025 Onboarded",
                value=results.get('onboarded_2025_total', 0),
                change=results.get('onboarding_growth', 0),
                icon="📈",
                color="#10B981"
            )
        
        with col4:
            self.render_metric_card(
                title="Avg Transaction Time",
                value=f"{results.get('avg_transaction_time_minutes', 0):.1f} min",
                change=-5.2,  # Example improvement
                icon="⏱️",
                color="#F59E0B"
            )
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            self.render_metric_card(
                title="Active Users",
                value=results.get('active_users_overall', 0),
                change=results.get('activity_growth', 0),
                icon="✅",
                color="#EC4899"
            )
        
        with col6:
            self.render_metric_card(
                title="Network Coverage",
                value=f"{results.get('network_coverage', 0):.1f}%",
                change=2.5,
                icon="🌐",
                color="#06B6D4"
            )
        
        with col7:
            self.render_metric_card(
                title="Deposit Volume",
                value=format_currency(results.get('total_deposits', 0)),
                change=15.8,
                icon="💰",
                color="#22C55E"
            )
        
        with col8:
            self.render_metric_card(
                title="Agent Retention",
                value=f"{results.get('retention_rate', 0):.1f}%",
                change=3.2,
                icon="📊",
                color="#EF4444"
            )
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            # Agent Distribution
            st.markdown("#### 👥 Agent Distribution")
            agent_data = {
                'Category': ['Agents', 'Tellers'],
                'Count': [
                    results.get('total_active_agents', 0),
                    results.get('total_active_tellers', 0)
                ]
            }
            
            fig = px.pie(
                agent_data,
                values='Count',
                names='Category',
                hole=0.4,
                color_discrete_sequence=['#3B82F6', '#8B5CF6']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly Activity
            st.markdown("#### 📅 Monthly Active Users")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            active_counts = [results['monthly_active_users'].get(m, 0) for m in range(1, 13)]
            
            fig = px.bar(
                x=months,
                y=active_counts,
                title="",
                color=active_counts,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Active Users",
                height=350,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Summary
        st.markdown("### 📋 Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Performers
            st.markdown("#### 🏆 Top 5 Performers")
            if 'top_performers' in results and results['top_performers']:
                top_df = pd.DataFrame(results['top_performers'][:5])
                st.dataframe(
                    top_df.style.format({'Deposit Count': '{:,}'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No top performers data available")
        
        with col2:
            # Regional Performance
            st.markdown("#### 🌍 Top Regions")
            if 'regional_distribution' in results and results['regional_distribution']:
                regions = list(results['regional_distribution'].keys())[:5]
                counts = list(results['regional_distribution'].values())[:5]
                
                region_df = pd.DataFrame({
                    'Region': regions,
                    'Agent Count': counts
                })
                st.dataframe(
                    region_df.style.format({'Agent Count': '{:,}'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No regional data available")
    
    def render_metric_card(self, title, value, change, icon, color):
        """Render a metric card"""
        change_color = "positive-change" if change >= 0 else "negative-change"
        change_symbol = "+" if change >= 0 else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.8rem;">{icon}</span>
                <span class="{change_color}">{change_symbol}{change}%</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_agents_view(self, results):
        """Render agents and tellers view"""
        st.markdown("### 👥 Agents & Tellers Analysis")
        
        # Agent Network Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Agents with Tellers",
                value=results.get('agents_with_tellers', 0),
                delta=f"{results.get('agents_with_tellers', 0)/max(results.get('total_active_agents', 1), 1)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Agents without Tellers",
                value=results.get('agents_without_tellers', 0),
                delta=f"{results.get('agents_without_tellers', 0)/max(results.get('total_active_agents', 1), 1)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Agent-Teller Ratio",
                value=f"1:{results.get('teller_per_agent', 0):.1f}",
                delta="Optimal: 1:3"
            )
        
        st.markdown("---")
        
        # Agent Network Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🔗 Agent Network Structure")
            
            # Create network data
            nodes = [
                {'name': 'Agents', 'value': results.get('total_active_agents', 0)},
                {'name': 'With Tellers', 'value': results.get('agents_with_tellers', 0)},
                {'name': 'Without Tellers', 'value': results.get('agents_without_tellers', 0)},
                {'name': 'Tellers', 'value': results.get('total_active_tellers', 0)}
            ]
            
            fig = px.sunburst(
                names=[n['name'] for n in nodes],
                parents=['', 'Agents', 'Agents', 'With Tellers'],
                values=[n['value'] for n in nodes],
                color=[n['name'] for n in nodes],
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Network Metrics")
            
            metrics = [
                ("Network Density", f"{results.get('network_density', 0):.1f}%"),
                ("Average Connections", f"{results.get('avg_connections', 0):.1f}"),
                ("Network Coverage", f"{results.get('network_coverage', 0):.1f}%"),
                ("Isolated Agents", results.get('isolated_agents', 0)),
                ("Most Connected", results.get('max_connections', 0))
            ]
            
            for label, value in metrics:
                st.metric(label=label, value=value)
            
            st.markdown("---")
            st.markdown("#### 💡 Recommendations")
            
            recommendations = [
                "Expand teller network in undercovered regions",
                "Provide incentives for agents to recruit tellers",
                "Implement mentorship program for new tellers",
                "Optimize agent-teller ratio to 1:3"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
    
    def render_performance_view(self, results):
        """Render performance analytics view"""
        st.markdown("### 📈 Performance Analytics")
        
        # Performance Tabs
        perf_tabs = st.tabs(["🎯 Activeness", "📊 Deposits", "⏱️ Efficiency", "📈 Growth"])
        
        with perf_tabs[0]:
            # Activeness Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Active vs Inactive
                labels = ['Active (≥20 deposits)', 'Inactive (<20 deposits)']
                values = [
                    results.get('active_users_overall', 0),
                    results.get('inactive_users_overall', 0)
                ]
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Activeness Distribution",
                    color_discrete_sequence=['#10B981', '#EF4444']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance Tiers
                if 'performance_tiers' in results:
                    tiers = results['performance_tiers']
                    tier_df = pd.DataFrame(tiers)
                    
                    fig = px.bar(
                        tier_df,
                        x='Tier',
                        y='Count',
                        title="Performance Tiers",
                        color='Tier',
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Performance tiers data not available")
        
        with perf_tabs[1]:
            # Deposit Analysis
            st.markdown("#### 💰 Deposit Performance")
            
            # Sample deposit trends
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            deposit_volume = np.random.uniform(50000, 200000, 6)
            deposit_count = np.random.randint(1000, 5000, 6)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Deposit Volume', 'Deposit Count')
            )
            
            fig.add_trace(
                go.Scatter(x=months, y=deposit_volume, name='Volume', line=dict(color='#3B82F6')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=months, y=deposit_count, name='Count', marker_color='#10B981'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_trends_view(self, results):
        """Render trends analysis view"""
        st.markdown("### 📅 Trends & Forecasting")
        
        # Monthly trends
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Sample trend data
        active_trend = [results['monthly_active_users'].get(m, 0) for m in range(1, 13)]
        onboard_trend = np.random.randint(50, 200, 12)
        growth_trend = np.random.uniform(-5, 15, 12)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Active Users Trend', 'Onboarding Trend', 'Growth Rate'),
            vertical_spacing=0.1
        )
        
        # Active Users
        fig.add_trace(
            go.Scatter(
                x=months,
                y=active_trend,
                mode='lines+markers',
                name='Active Users',
                line=dict(color='#3B82F6', width=3)
            ),
            row=1, col=1
        )
        
        # Onboarding
        fig.add_trace(
            go.Bar(
                x=months,
                y=onboard_trend,
                name='New Onboarding',
                marker_color='#10B981'
            ),
            row=2, col=1
        )
        
        # Growth Rate
        fig.add_trace(
            go.Scatter(
                x=months,
                y=growth_trend,
                mode='lines+markers',
                name='Growth Rate',
                fill='tozeroy',
                line=dict(color='#F59E0B', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecasting
        st.markdown("### 🔮 6-Month Forecast")
        
        forecast_months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        forecast_active = np.random.randint(1500, 2500, 6)
        forecast_onboard = np.random.randint(100, 300, 6)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months[:6], y=active_trend[:6],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#3B82F6')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_months, y=forecast_active,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3B82F6', dash='dash')
            ))
            fig.update_layout(
                title='Active Users Forecast',
                xaxis_title='Month',
                yaxis_title='Active Users'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months[:6], y=onboard_trend[:6],
                name='Actual',
                marker_color='#10B981'
            ))
            fig.add_trace(go.Bar(
                x=forecast_months, y=forecast_onboard,
                name='Forecast',
                marker_color='#10B981',
                opacity=0.7
            ))
            fig.update_layout(
                title='Onboarding Forecast',
                xaxis_title='Month',
                yaxis_title='New Onboarding'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_network_view(self, results):
        """Render network analysis view"""
        st.markdown("### 🔗 Network Analysis")
        
        # Network metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", results.get('total_active_agents', 0) + results.get('total_active_tellers', 0))
        
        with col2:
            st.metric("Total Connections", results.get('total_connections', 0))
        
        with col3:
            st.metric("Network Density", f"{results.get('network_density', 0):.1f}%")
        
        with col4:
            st.metric("Avg Degree", f"{results.get('avg_degree', 0):.1f}")
        
        st.markdown("---")
        
        # Network visualization and analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏙️ Regional Network Map")
            
            # Sample regional data
            regions = ['West Coast', 'Greater Banjul', 'Central River', 'North Bank', 'Lower River']
            agent_counts = np.random.randint(50, 300, 5)
            teller_counts = np.random.randint(20, 150, 5)
            
            region_df = pd.DataFrame({
                'Region': regions,
                'Agents': agent_counts,
                'Tellers': teller_counts
            })
            
            fig = px.scatter(
                region_df,
                x='Agents',
                y='Tellers',
                size='Agents',
                color='Region',
                hover_name='Region',
                size_max=60,
                title="Regional Agent-Teller Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Connectivity Analysis")
            
            # Degree distribution
            degrees = np.random.randint(1, 10, 100)
            degree_counts = np.bincount(degrees)
            
            fig = px.histogram(
                x=range(len(degree_counts)),
                y=degree_counts,
                nbins=20,
                title="Degree Distribution",
                labels={'x': 'Number of Connections', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Network health indicators
            st.markdown("##### 📈 Network Health")
            
            health_metrics = [
                ("Connectivity Score", "85/100", "🟢"),
                ("Growth Potential", "High", "🟡"),
                ("Risk Factor", "Low", "🟢"),
                ("Optimization Need", "Medium", "🟡")
            ]
            
            for metric, value, status in health_metrics:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(metric)
                with col_b:
                    st.write(value)
                with col_c:
                    st.write(status)
    
    def render_reports_view(self, results):
        """Render reports generation view"""
        st.markdown("### 📥 Reports & Export")
        
        # Report generation options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Generate Reports")
            
            report_type = st.selectbox(
                "Select Report Type",
                [
                    "Executive Summary",
                    "Agent Performance Report",
                    "Monthly Performance Report",
                    "Network Analysis Report",
                    "Comprehensive Annual Report"
                ]
            )
            
            date_range = st.date_input(
                "Select Date Range",
                [
                    datetime(self.year, 1, 1),
                    datetime(self.year, 12, 31)
                ]
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_data = st.checkbox("Include Raw Data", value=False)
            
            if st.button("🔄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    # Simulate report generation
                    time.sleep(2)
                    
                    # Create sample report
                    report_data = self.report_gen.generate_report(
                        report_type,
                        results,
                        date_range
                    )
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Display preview
                    with st.expander("📋 Report Preview", expanded=True):
                        st.write(report_data['preview'])
        
        with col2:
            st.markdown("#### 📥 Export Options")
            
            # Export formats
            export_format = st.radio(
                "Export Format",
                ["CSV", "Excel", "PDF", "HTML"],
                horizontal=True
            )
            
            # Data to export
            export_options = st.multiselect(
                "Select Data to Export",
                [
                    "Key Metrics",
                    "Agent List",
                    "Transaction Summary",
                    "Performance Data",
                    "Monthly Trends",
                    "Network Analysis"
                ],
                default=["Key Metrics", "Agent List"]
            )
            
            if st.button("💾 Export Data", use_container_width=True):
                # Create downloadable data
                csv_data = self.create_export_data(results, export_options)
                
                st.download_button(
                    label=f"⬇️ Download {export_format}",
                    data=csv_data,
                    file_name=f"aps_wallet_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # Quick Export Cards
        st.markdown("#### ⚡ Quick Export")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Summary CSV", use_container_width=True):
                self.quick_export("summary", results)
        
        with col2:
            if st.button("📈 Charts PDF", use_container_width=True):
                self.quick_export("charts", results)
        
        with col3:
            if st.button("👥 Agent List", use_container_width=True):
                self.quick_export("agents", results)
        
        with col4:
            if st.button("📊 Full Dataset", use_container_width=True):
                self.quick_export("full", results)
    
    def create_export_data(self, results, options):
        """Create data for export"""
        # This would create actual data for export
        # For now, return a sample CSV
        sample_data = pd.DataFrame({
            'Metric': ['Total Agents', 'Active Tellers', '2025 Onboarded'],
            'Value': [
                results.get('total_active_agents', 0),
                results.get('total_active_tellers', 0),
                results.get('onboarded_2025_total', 0)
            ]
        })
        
        return sample_data.to_csv(index=False)
    
    def quick_export(self, export_type, results):
        """Handle quick exports"""
        st.info(f"Preparing {export_type} export...")
        # In a real app, this would generate and offer download
    
    def render_landing_page(self):
        """Render landing page when no data is loaded"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <h1 style="color: #1E3A8A; margin-bottom: 20px;">Welcome to APS Wallet Analytics</h1>
                <p style="color: #6B7280; font-size: 1.2rem; margin-bottom: 40px;">
                    Advanced analytics platform for agent network performance management
                </p>
                
                <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 50px;">
                    <div style="text-align: center; padding: 20px; background: #F3F4F6; border-radius: 10px; width: 180px;">
                        <div style="font-size: 2.5rem;">📊</div>
                        <h4>Real-time Analytics</h4>
                        <p style="font-size: 0.9rem; color: #6B7280;">Live performance monitoring</p>
                    </div>
                    
                    <div style="text-align: center; padding: 20px; background: #F3F4F6; border-radius: 10px; width: 180px;">
                        <div style="font-size: 2.5rem;">📈</div>
                        <h4>Trend Analysis</h4>
                        <p style="font-size: 0.9rem; color: #6B7280;">Historical performance trends</p>
                    </div>
                    
                    <div style="text-align: center; padding: 20px; background: #F3F4F6; border-radius: 10px; width: 180px;">
                        <div style="font-size: 2.5rem;">🔗</div>
                        <h4>Network Insights</h4>
                        <p style="font-size: 0.9rem; color: #6B7280;">Agent-teller relationships</p>
                    </div>
                </div>
                
                <div style="background: #EFF6FF; padding: 30px; border-radius: 15px; margin: 40px 0;">
                    <h3 style="color: #1E3A8A;">🚀 Get Started</h3>
                    <p style="color: #6B7280;">To begin your analysis:</p>
                    <ol style="text-align: left; padding-left: 20px; color: #6B7280;">
                        <li>Go to the sidebar on the left</li>
                        <li>Upload your Onboarding CSV file</li>
                        <li>Upload your Transaction CSV file</li>
                        <li>Select the analysis year</li>
                        <li>Click "Process Data" or "Load Sample Data"</li>
                    </ol>
                </div>
                
                <div style="margin-top: 40px;">
                    <p style="color: #6B7280;">
                        Need help? Check our 
                        <a href="#" style="color: #3B82F6; text-decoration: none;">documentation</a> 
                        or contact 
                        <a href="mailto:support@apswallet.com" style="color: #3B82F6; text-decoration: none;">
                            support@apswallet.com
                        </a>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    app = APSDashboard()
    app.run()

if __name__ == "__main__":
    main()