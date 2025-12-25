"""
Main Dashboard Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard | APS Wallet", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 APS Wallet Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time Performance Monitoring • {date}</p>
    </div>
    """.format(date=datetime.now().strftime("%d %B %Y")), unsafe_allow_html=True)
    
    # Check if data is loaded
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("⚠️ No data loaded. Please upload data from the main page or load sample data.")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("📋 Load Sample Dashboard", type="primary", use_container_width=True):
                # Load sample data
                sample_data = generate_sample_data()
                st.session_state.analysis_results = sample_data
                st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    # Key Performance Indicators
    st.markdown("## 🎯 Key Performance Indicators")
    
    # Row 1 - Primary KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{results.get('total_active_agents', 0):,}</div>
            <div class="kpi-label">Active Agents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{results.get('total_active_tellers', 0):,}</div>
            <div class="kpi-label">Agent Tellers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{results.get('active_users_overall', 0):,}</div>
            <div class="kpi-label">Active Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        growth = results.get('agent_growth', 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{growth:.1f}%</div>
            <div class="kpi-label">Growth Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2 - Secondary KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{results.get('onboarded_2025_total', 0):,}</div>
            <div class="kpi-label">2025 Onboarded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        coverage = results.get('network_coverage', 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{coverage:.1f}%</div>
            <div class="kpi-label">Network Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{results.get('avg_transaction_time_minutes', 0):.1f} min</div>
            <div class="kpi-label">Avg Transaction Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        retention = results.get('retention_rate', 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{retention:.1f}%</div>
            <div class="kpi-label">Retention Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Charts Section
    st.markdown("## 📈 Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent Distribution
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
            title='Agent vs Teller Distribution',
            hole=0.4,
            color_discrete_sequence=['#3B82F6', '#8B5CF6']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Network Status
        st.markdown("#### 🔗 Network Status")
        network_data = {
            'Metric': ['Density', 'Connections', 'Avg Degree'],
            'Value': [
                f"{results.get('network_density', 0):.1f}%",
                results.get('total_connections', 0),
                f"{results.get('avg_degree', 0):.1f}"
            ]
        }
        st.dataframe(pd.DataFrame(network_data), use_container_width=True, hide_index=True)
    
    with col2:
        # Monthly Active Users
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        active_counts = [results['monthly_active_users'].get(m, 0) for m in range(1, 13)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=months,
                y=active_counts,
                marker_color='#10B981',
                text=active_counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Monthly Active Users (≥20 deposits)',
            xaxis_title='Month',
            yaxis_title='Active Users',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Summary
        st.markdown("#### 📊 Performance Summary")
        summary_data = {
            'Category': ['Active', 'Inactive', 'Total'],
            'Users': [
                results.get('active_users_overall', 0),
                results.get('inactive_users_overall', 0),
                results.get('active_users_overall', 0) + results.get('inactive_users_overall', 0)
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Regional Analysis (if available)
    if 'regional_distribution' in results and results['regional_distribution']:
        st.markdown("## 🌍 Regional Analysis")
        
        regions = list(results['regional_distribution'].keys())
        counts = list(results['regional_distribution'].values())
        
        fig = px.bar(
            x=regions,
            y=counts,
            title='Top Regions by Agent Count',
            labels={'x': 'Region', 'y': 'Agent Count'},
            color=counts,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick Actions
    st.markdown("---")
    st.markdown("## ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Dashboard", use_container_width=True):
            st.success("Dashboard exported successfully!")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.info("Data refresh initiated...")
    
    with col3:
        if st.button("📧 Share Report", use_container_width=True):
            st.info("Report sharing options opened...")

def generate_sample_data():
    """Generate sample data for dashboard"""
    return {
        'year': 2025,
        'total_active_agents': 1798,
        'total_active_tellers': 1370,
        'active_users_overall': 1954,
        'inactive_users_overall': 861,
        'agents_with_tellers': 864,
        'agents_without_tellers': 934,
        'onboarded_2025_total': 2806,
        'onboarded_2025_agents': 1730,
        'onboarded_2025_tellers': 1076,
        'avg_transaction_time_minutes': 2.5,
        'agent_growth': 15.8,
        'network_coverage': 78.5,
        'retention_rate': 82.3,
        'network_density': 45.2,
        'total_connections': 12500,
        'avg_degree': 3.2,
        'monthly_active_users': {m: np.random.randint(100, 300) for m in range(1, 13)},
        'regional_distribution': {
            'West Coast': 450,
            'Greater Banjul': 380,
            'Central River': 320,
            'North Bank': 280,
            'Lower River': 220,
            'Upper River': 180
        }
    }

if __name__ == "__main__":
    main()