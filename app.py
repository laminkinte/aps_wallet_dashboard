"""
APS Wallet - Annual Agent Performance Dashboard
Simplified version with compatible dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
import io
import base64
import re

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="APS Wallet | Agent Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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

class AgentPerformanceAnalyzer:
    def __init__(self):
        self.year = 2025
        self.df_onboarding = None
        self.df_transactions = None
        self.results = None
    
    def load_data(self, onboarding_file, transaction_file):
        """Load and process data"""
        with st.spinner('Loading and processing data...'):
            try:
                # Load data
                self.df_onboarding = pd.read_csv(onboarding_file)
                self.df_transactions = pd.read_csv(transaction_file, nrows=100000)  # Limit for performance
                
                # Clean data
                self._clean_data()
                
                # Calculate metrics
                self.results = self._calculate_metrics()
                
                return True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
    
    def _clean_data(self):
        """Clean and preprocess data"""
        # Clean onboarding data
        self.df_onboarding['Entity'] = self.df_onboarding['Entity'].astype(str).str.upper().str.strip()
        self.df_onboarding['Status'] = self.df_onboarding['Status'].astype(str).str.upper().str.strip()
        
        # Parse registration date
        self.df_onboarding['Registration Date'] = pd.to_datetime(
            self.df_onboarding['Registration Date'], 
            format='%d/%m/%Y %H:%M', 
            errors='coerce'
        )
        
        # Clean transaction data
        self.df_transactions['Created At'] = pd.to_datetime(
            self.df_transactions['Created At'], 
            errors='coerce'
        )
        
        # Extract year and month
        self.df_transactions['Year'] = self.df_transactions['Created At'].dt.year
        self.df_transactions['Month'] = self.df_transactions['Created At'].dt.month
        
        # Clean identifiers
        def clean_identifier(x):
            if pd.isna(x):
                return np.nan
            try:
                # Remove non-numeric characters
                cleaned = re.sub(r'[^\d.]', '', str(x))
                return int(float(cleaned)) if cleaned else np.nan
            except:
                return np.nan
        
        if 'User Identifier' in self.df_transactions.columns:
            self.df_transactions['User Identifier'] = self.df_transactions['User Identifier'].apply(clean_identifier)
        
        if 'Parent User Identifier' in self.df_transactions.columns:
            self.df_transactions['Parent User Identifier'] = self.df_transactions['Parent User Identifier'].apply(clean_identifier)
        
        # Clean text columns
        text_cols = ['Entity Name', 'Service Name', 'Transaction Type', 'Product Name']
        for col in text_cols:
            if col in self.df_transactions.columns:
                self.df_transactions[col] = self.df_transactions[col].astype(str).str.upper().str.strip()
    
    def _calculate_metrics(self):
        """Calculate all performance metrics"""
        results = {
            'year': self.year,
            'total_active_agents': 0,
            'total_active_tellers': 0,
            'agents_with_tellers': 0,
            'agents_without_tellers': 0,
            'onboarded_2025_total': 0,
            'onboarded_2025_agents': 0,
            'onboarded_2025_tellers': 0,
            'active_users_overall': 0,
            'inactive_users_overall': 0,
            'monthly_active_users': {m: 0 for m in range(1, 13)},
            'avg_transaction_time_minutes': 0.0
        }
        
        # Filter for 2025 transactions
        df_transactions_2025 = self.df_transactions[self.df_transactions['Year'] == self.year].copy()
        
        # 1. & 2. Active Agents and Tellers
        terminated_status = {'TERMINATED', 'BLOCKED', 'SUSPENDED', 'INACTIVE'}
        active_mask = ~self.df_onboarding['Status'].isin(terminated_status)
        df_active = self.df_onboarding[active_mask]
        
        results['total_active_agents'] = len(df_active[df_active['Entity'] == 'AGENT'])
        results['total_active_tellers'] = len(df_active[df_active['Entity'] == 'AGENT TELLER'])
        
        # 3. & 4. Agents with/without Tellers
        if 'Parent User Identifier' in df_transactions_2025.columns:
            parent_ids = df_transactions_2025['Parent User Identifier'].dropna().unique()
            results['agents_with_tellers'] = len(parent_ids)
            
            # For demonstration - calculate agents without tellers
            total_agents = results['total_active_agents']
            results['agents_without_tellers'] = max(0, total_agents - results['agents_with_tellers'])
        
        # 5. Onboarding in 2025
        mask_2025_onboarding = (self.df_onboarding['Registration Date'].dt.year == self.year) & active_mask
        df_onboarded_2025 = self.df_onboarding[mask_2025_onboarding]
        
        results['onboarded_2025_total'] = len(df_onboarded_2025)
        results['onboarded_2025_agents'] = len(df_onboarded_2025[df_onboarded_2025['Entity'] == 'AGENT'])
        results['onboarded_2025_tellers'] = len(df_onboarded_2025[df_onboarded_2025['Entity'] == 'AGENT TELLER'])
        
        # 6. & 7. Activeness analysis
        if not df_transactions_2025.empty:
            # Identify deposits
            deposit_mask = (
                df_transactions_2025['Service Name'].str.contains('DEPOSIT', na=False) |
                df_transactions_2025['Transaction Type'].str.contains('DEPOSIT', na=False) |
                df_transactions_2025['Product Name'].str.contains('DEPOSIT', na=False)
            )
            
            df_deposits = df_transactions_2025[deposit_mask]
            
            if not df_deposits.empty:
                # Filter for agents/tellers
                agent_mask = df_deposits['Entity Name'].isin(['AGENT', 'AGENT TELLER'])
                df_agent_deposits = df_deposits[agent_mask]
                
                if 'User Identifier' in df_agent_deposits.columns:
                    # Overall activeness
                    deposit_counts = df_agent_deposits['User Identifier'].value_counts()
                    results['active_users_overall'] = len(deposit_counts[deposit_counts >= 20])
                    results['inactive_users_overall'] = len(deposit_counts[deposit_counts < 20])
                    
                    # Monthly activeness
                    for month in range(1, 13):
                        month_mask = df_agent_deposits['Month'] == month
                        month_deposits = df_agent_deposits[month_mask]
                        if not month_deposits.empty:
                            month_counts = month_deposits['User Identifier'].value_counts()
                            results['monthly_active_users'][month] = len(month_counts[month_counts >= 20])
        
        # 8. Transaction time (simplified calculation)
        if not df_transactions_2025.empty:
            # Sample transactions for time calculation
            sample_size = min(1000, len(df_transactions_2025))
            df_sample = df_transactions_2025.sample(n=sample_size)
            
            if len(df_sample) > 1:
                df_sample = df_sample.sort_values('Created At')
                time_diffs = df_sample['Created At'].diff().dt.total_seconds().mean() / 60
                results['avg_transaction_time_minutes'] = round(time_diffs, 2)
        
        return results

def main():
    """Main Streamlit application"""
    st.title("📊 APS Wallet - Agent Performance Dashboard 2025")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = AgentPerformanceAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wallet--v1.png", width=80)
        st.markdown("## 📤 Upload Data Files")
        
        # File uploaders
        onboarding_file = st.file_uploader(
            "Upload Onboarding CSV",
            type=['csv'],
            help="Upload the onboarding data file"
        )
        
        transaction_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload the transaction data file"
        )
        
        st.markdown("---")
        
        # Load sample data button
        if st.button("📋 Load Sample Data", use_container_width=True):
            # Create sample data
            sample_onboarding = pd.DataFrame({
                'Account ID': [f'ACC-{i:06d}' for i in range(1, 101)],
                'Entity': np.random.choice(['AGENT', 'AGENT TELLER'], 100),
                'Status': np.random.choice(['ACTIVE', 'TERMINATED', 'INACTIVE'], 100, p=[0.8, 0.1, 0.1]),
                'Registration Date': pd.date_range('2024-01-01', periods=100),
                'Region': np.random.choice(['West Coast', 'Greater Banjul', 'Central River'], 100)
            })
            
            sample_transactions = pd.DataFrame({
                'User Identifier': np.random.randint(1000, 2000, 10000),
                'Parent User Identifier': np.random.choice([1001, 1002, 1003, np.nan], 10000),
                'Entity Name': np.random.choice(['AGENT', 'AGENT TELLER', 'CUSTOMER'], 10000),
                'Service Name': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], 10000),
                'Transaction Type': np.random.choice(['CR', 'DR'], 10000),
                'Created At': pd.date_range('2025-01-01', periods=10000, freq='H')
            })
            
            # Save to session state
            st.session_state.sample_onboarding = sample_onboarding
            st.session_state.sample_transactions = sample_transactions
            st.success("Sample data loaded!")
    
    # Check if we have data to analyze
    data_loaded = False
    results = None
    
    if onboarding_file and transaction_file:
        # Analyze uploaded data
        with st.spinner("Analyzing data..."):
            if analyzer.load_data(onboarding_file, transaction_file):
                data_loaded = True
                results = analyzer.results
    elif 'sample_onboarding' in st.session_state and 'sample_transactions' in st.session_state:
        # Analyze sample data
        analyzer.df_onboarding = st.session_state.sample_onboarding
        analyzer.df_transactions = st.session_state.sample_transactions
        analyzer._clean_data()
        results = analyzer._calculate_metrics()
        data_loaded = True
    
    if data_loaded and results:
        # Display metrics
        st.markdown("## 📊 Key Performance Indicators")
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Agents", f"{results['total_active_agents']:,}")
        
        with col2:
            st.metric("Agent Tellers", f"{results['total_active_tellers']:,}")
        
        with col3:
            st.metric("Active Users", f"{results['active_users_overall']:,}")
        
        with col4:
            st.metric("Avg Transaction Time", f"{results['avg_transaction_time_minutes']:.1f} min")
        
        st.markdown("---")
        
        # Detailed analysis
        st.markdown("## 📈 Detailed Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Agents & Tellers", "📅 Monthly Activity", "📊 Performance", "📥 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent distribution pie chart
                fig = px.pie(
                    values=[results['total_active_agents'], results['total_active_tellers']],
                    names=['Agents', 'Tellers'],
                    title='Agent vs Teller Distribution',
                    color_discrete_sequence=['#3B82F6', '#8B5CF6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Agent-teller relationships
                st.markdown("### Agent-Teller Network")
                relationship_data = {
                    'Category': ['Agents with Tellers', 'Agents without Tellers'],
                    'Count': [results['agents_with_tellers'], results['agents_without_tellers']]
                }
                df_relationships = pd.DataFrame(relationship_data)
                st.dataframe(df_relationships, use_container_width=True)
        
        with tab2:
            # Monthly activity chart
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            active_counts = [results['monthly_active_users'][m] for m in range(1, 13)]
            
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
        
        with tab3:
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Activeness Analysis")
                activeness_data = {
                    'Status': ['Active (≥20 deposits)', 'Inactive (<20 deposits)'],
                    'Users': [results['active_users_overall'], results['inactive_users_overall']]
                }
                df_activeness = pd.DataFrame(activeness_data)
                st.dataframe(df_activeness, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 2025 Onboarding")
                onboarding_data = {
                    'Type': ['Agents', 'Tellers', 'Total'],
                    'Count': [
                        results['onboarded_2025_agents'],
                        results['onboarded_2025_tellers'],
                        results['onboarded_2025_total']
                    ]
                }
                df_onboarding = pd.DataFrame(onboarding_data)
                st.dataframe(df_onboarding, use_container_width=True)
        
        with tab4:
            # Export section
            st.markdown("### 📥 Export Results")
            
            # Create export data
            summary_data = pd.DataFrame({
                'Metric': [
                    'Total Active Agents',
                    'Total Active Agent Tellers',
                    'Agents with Agent Tellers',
                    'Agents without Agent Tellers',
                    'Total Onboarded 2025',
                    'Agents Onboarded 2025',
                    'Agent Tellers Onboarded 2025',
                    'Active Users Overall',
                    'Inactive Users Overall',
                    'Average Transaction Time (minutes)'
                ],
                'Value': [
                    results['total_active_agents'],
                    results['total_active_tellers'],
                    results['agents_with_tellers'],
                    results['agents_without_tellers'],
                    results['onboarded_2025_total'],
                    results['onboarded_2025_agents'],
                    results['onboarded_2025_tellers'],
                    results['active_users_overall'],
                    results['inactive_users_overall'],
                    results['avg_transaction_time_minutes']
                ]
            })
            
            # Convert to CSV
            csv = summary_data.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=csv,
                file_name=f"aps_wallet_summary_{results['year']}.csv",
                mime="text/csv"
            )
            
            # Monthly data export
            monthly_data = pd.DataFrame([
                {'Month': datetime(results['year'], m, 1).strftime('%B'), 
                 'Active_Users': results['monthly_active_users'][m]}
                for m in range(1, 13)
            ])
            
            monthly_csv = monthly_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Monthly Report (CSV)",
                data=monthly_csv,
                file_name=f"aps_wallet_monthly_{results['year']}.csv",
                mime="text/csv"
            )
    
    else:
        # Show welcome message
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="color: #1E3A8A;">Welcome to APS Wallet Analytics</h1>
            <p style="color: #6B7280; font-size: 1.2rem;">
                Advanced analytics platform for agent network performance management
            </p>
            
            <div style="background: #EFF6FF; padding: 30px; border-radius: 15px; margin: 40px 0;">
                <h3 style="color: #1E3A8A;">🚀 Get Started</h3>
                <p>To begin your analysis:</p>
                <ol style="text-align: left; padding-left: 20px;">
                    <li>Upload your Onboarding CSV file</li>
                    <li>Upload your Transaction CSV file</li>
                    <li>Or click "Load Sample Data" to use demo data</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
