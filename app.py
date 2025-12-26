"""
APS Wallet - Agent Performance Dashboard 2025
Minimal, working version for Python 3.13
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="APS Wallet Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #3B82F6;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

def analyze_data(df_onboarding, df_transactions, year=2025):
    """Simple analysis function"""
    results = {
        'year': year,
        'total_active_agents': 0,
        'total_active_tellers': 0,
        'agents_with_tellers': 0,
        'agents_without_tellers': 0,
        'onboarded_total': 0,
        'active_users': 0,
        'inactive_users': 0,
        'monthly_active': {},
        'avg_transaction_time': 0
    }
    
    try:
        # Basic agent counts
        if 'Entity' in df_onboarding.columns:
            results['total_active_agents'] = len(df_onboarding[df_onboarding['Entity'].str.contains('AGENT', na=False)])
            results['total_active_tellers'] = len(df_onboarding[df_onboarding['Entity'].str.contains('TELLER', na=False)])
        
        # 2025 onboarding
        if 'Registration Date' in df_onboarding.columns:
            df_onboarding['Registration Date'] = pd.to_datetime(df_onboarding['Registration Date'], errors='coerce')
            mask_2025 = df_onboarding['Registration Date'].dt.year == year
            results['onboarded_total'] = len(df_onboarding[mask_2025])
        
        # Transaction analysis
        if not df_transactions.empty:
            df_transactions['Created At'] = pd.to_datetime(df_transactions['Created At'], errors='coerce')
            df_transactions['Year'] = df_transactions['Created At'].dt.year
            df_transactions['Month'] = df_transactions['Created At'].dt.month
            
            df_2025 = df_transactions[df_transactions['Year'] == year].copy()
            
            # Deposit identification
            deposit_cols = ['Service Name', 'Transaction Type', 'Product Name']
            for col in deposit_cols:
                if col in df_2025.columns:
                    df_2025[col] = df_2025[col].astype(str).str.upper()
            
            deposit_mask = (
                df_2025['Service Name'].str.contains('DEPOSIT', na=False) |
                df_2025['Transaction Type'].str.contains('DEPOSIT', na=False) |
                df_2025['Product Name'].str.contains('DEPOSIT', na=False)
            )
            
            df_deposits = df_2025[deposit_mask]
            
            if 'User Identifier' in df_deposits.columns:
                # Clean user identifiers
                df_deposits['User Identifier'] = pd.to_numeric(
                    df_deposits['User Identifier'].astype(str).str.replace(r'\D', '', regex=True),
                    errors='coerce'
                )
                
                # Count deposits per user
                deposit_counts = df_deposits['User Identifier'].value_counts()
                results['active_users'] = len(deposit_counts[deposit_counts >= 20])
                results['inactive_users'] = len(deposit_counts[deposit_counts < 20])
                
                # Monthly activity
                for month in range(1, 13):
                    month_mask = df_deposits['Month'] == month
                    month_counts = df_deposits[month_mask]['User Identifier'].value_counts()
                    results['monthly_active'][month] = len(month_counts[month_counts >= 20])
                
                # Transaction time (simplified)
                if len(df_deposits) > 100:
                    df_sample = df_deposits.sample(n=min(1000, len(df_deposits)))
                    df_sample = df_sample.sort_values('Created At')
                    time_diffs = df_sample['Created At'].diff().dt.total_seconds() / 60
                    valid_times = time_diffs.between(0.1, 30)
                    if valid_times.any():
                        results['avg_transaction_time'] = round(time_diffs[valid_times].mean(), 2)
        
        # Agents with tellers
        if 'Parent User Identifier' in df_transactions.columns:
            parent_ids = df_transactions['Parent User Identifier'].dropna().unique()
            results['agents_with_tellers'] = len(parent_ids)
            results['agents_without_tellers'] = max(0, results['total_active_agents'] - results['agents_with_tellers'])
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return results

def main():
    st.title("📊 APS Wallet - Agent Performance Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wallet--v1.png", width=80)
        st.markdown("## 📤 Upload Data")
        
        onboarding_file = st.file_uploader("Onboarding CSV", type=['csv'])
        transaction_file = st.file_uploader("Transaction CSV", type=['csv'])
        
        if st.button("📋 Use Sample Data", use_container_width=True):
            # Create sample data
            sample_onboarding = pd.DataFrame({
                'Account ID': [f'ACC{i:06d}' for i in range(1, 51)],
                'Entity': ['AGENT'] * 30 + ['AGENT TELLER'] * 20,
                'Status': ['ACTIVE'] * 45 + ['INACTIVE'] * 5,
                'Registration Date': pd.date_range('2024-01-01', periods=50),
                'Region': ['West Coast'] * 20 + ['Greater Banjul'] * 20 + ['Central River'] * 10
            })
            
            sample_transactions = pd.DataFrame({
                'User Identifier': np.random.randint(1000, 1500, 1000),
                'Parent User Identifier': np.random.choice([1001, 1002, 1003, 1004, np.nan], 1000),
                'Entity Name': np.random.choice(['AGENT', 'AGENT TELLER'], 1000),
                'Service Name': np.random.choice(['DEPOSIT', 'WITHDRAWAL'], 1000),
                'Transaction Type': ['CR'] * 500 + ['DR'] * 500,
                'Created At': pd.date_range('2025-01-01', periods=1000, freq='H')
            })
            
            st.session_state.df_onboarding = sample_onboarding
            st.session_state.df_transactions = sample_transactions
            st.session_state.year = 2025
            st.rerun()
    
    # Check for data
    data_available = False
    results = None
    
    if onboarding_file and transaction_file:
        try:
            df_onboarding = pd.read_csv(onboarding_file)
            df_transactions = pd.read_csv(transaction_file, nrows=100000)  # Limit for performance
            
            results = analyze_data(df_onboarding, df_transactions, 2025)
            data_available = True
            
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
    
    elif 'df_onboarding' in st.session_state:
        results = analyze_data(
            st.session_state.df_onboarding, 
            st.session_state.df_transactions,
            st.session_state.get('year', 2025)
        )
        data_available = True
    
    if data_available and results:
        # Display metrics
        st.markdown("## 📊 Performance Metrics 2025")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['total_active_agents']:,}</div>
                <div class="metric-label">Active Agents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['total_active_tellers']:,}</div>
                <div class="metric-label">Agent Tellers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['active_users']:,}</div>
                <div class="metric-label">Active Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['avg_transaction_time']:.1f}</div>
                <div class="metric-label">Avg Time (min)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['onboarded_total']:,}</div>
                <div class="metric-label">2025 Onboarded</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['agents_with_tellers']:,}</div>
                <div class="metric-label">Agents with Tellers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['inactive_users']:,}</div>
                <div class="metric-label">Inactive Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            coverage = (results['agents_with_tellers'] / max(results['total_active_agents'], 1)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{coverage:.1f}%</div>
                <div class="metric-label">Network Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📅 Monthly Activity", "📥 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent distribution
                fig = px.pie(
                    values=[results['total_active_agents'], results['total_active_tellers']],
                    names=['Agents', 'Tellers'],
                    title='Agent Distribution',
                    color_discrete_sequence=['#3B82F6', '#8B5CF6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Activeness
                fig = px.bar(
                    x=['Active', 'Inactive'],
                    y=[results['active_users'], results['inactive_users']],
                    title='User Activeness',
                    color=['Active', 'Inactive'],
                    color_discrete_sequence=['#10B981', '#EF4444']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Monthly activity
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            active_counts = [results['monthly_active'].get(m, 0) for m in range(1, 13)]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=months,
                    y=active_counts,
                    marker_color='#3B82F6',
                    text=active_counts
                )
            ])
            
            fig.update_layout(
                title='Monthly Active Users (≥20 deposits)',
                xaxis_title='Month',
                yaxis_title='Active Users',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Export section
            st.markdown("### 📥 Export Results")
            
            # Create summary dataframe
            summary_data = pd.DataFrame({
                'Metric': [
                    'Total Active Agents',
                    'Total Active Agent Tellers', 
                    'Agents with Agent Tellers',
                    'Agents without Agent Tellers',
                    '2025 Onboarded Total',
                    'Active Users (≥20 deposits)',
                    'Inactive Users (<20 deposits)',
                    'Average Transaction Time (minutes)'
                ],
                'Value': [
                    results['total_active_agents'],
                    results['total_active_tellers'],
                    results['agents_with_tellers'],
                    results['agents_without_tellers'],
                    results['onboarded_total'],
                    results['active_users'],
                    results['inactive_users'],
                    results['avg_transaction_time']
                ]
            })
            
            # Convert to CSV
            csv = summary_data.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download Summary Report",
                data=csv,
                file_name=f"aps_wallet_report_{results['year']}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Show preview
            st.dataframe(summary_data, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="color: #1E3A8A;">Welcome to APS Wallet Analytics</h1>
            <p style="color: #6B7280; font-size: 1.2rem;">
                Upload your data files to analyze agent performance for 2025
            </p>
            
            <div style="background: #EFF6FF; padding: 30px; border-radius: 15px; margin: 40px 0;">
                <h3 style="color: #1E3A8A;">📋 Required Analysis</h3>
                <ul style="text-align: left; color: #6B7280;">
                    <li>Total Agents & Agent Tellers (excluding terminated/blocked)</li>
                    <li>Agents with/without Agent Tellers</li>
                    <li>2025 Onboarding statistics</li>
                    <li>Activeness analysis (≥20 deposits monthly/annually)</li>
                    <li>Average transaction time</li>
                </ul>
            </div>
            
            <p style="color: #6B7280;">
                Upload your CSV files or click "Use Sample Data" to test the dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
