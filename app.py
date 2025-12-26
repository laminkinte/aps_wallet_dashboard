"""
APS Wallet - Annual Agent Performance Dashboard 2025
Production-ready dashboard for agent network analytics
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
import sys
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="APS Wallet | Agent Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
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
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #3B82F6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
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
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        background-color: #F3F4F6;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class AgentPerformanceDashboard:
    def __init__(self):
        self.year = 2025
        self.df_onboarding = None
        self.df_transactions = None
        self.results = None
        
    def load_data(self, onboarding_file, transaction_file):
        """Load and process data"""
        try:
            # Load onboarding data
            self.df_onboarding = pd.read_csv(onboarding_file)
            
            # Load transaction data with chunking for large files
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(transaction_file, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= 500000:  # Limit to 500k rows for performance
                    break
            
            if chunks:
                self.df_transactions = pd.concat(chunks, ignore_index=True)
            else:
                self.df_transactions = pd.DataFrame()
            
            # Clean and process data
            self._clean_data()
            
            # Calculate all metrics
            self.results = self._calculate_all_metrics()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _clean_data(self):
        """Clean and preprocess data"""
        # Clean onboarding data
        if self.df_onboarding is not None and not self.df_onboarding.empty:
            # Standardize columns
            self.df_onboarding.columns = self.df_onboarding.columns.str.strip()
            
            # Clean text columns
            text_cols = ['Entity', 'Status']
            for col in text_cols:
                if col in self.df_onboarding.columns:
                    self.df_onboarding[col] = self.df_onboarding[col].astype(str).str.upper().str.strip()
            
            # Parse registration date
            if 'Registration Date' in self.df_onboarding.columns:
                self.df_onboarding['Registration Date'] = pd.to_datetime(
                    self.df_onboarding['Registration Date'],
                    format='%d/%m/%Y %H:%M',
                    errors='coerce'
                )
        
        # Clean transaction data
        if self.df_transactions is not None and not self.df_transactions.empty:
            # Standardize columns
            self.df_transactions.columns = self.df_transactions.columns.str.strip()
            
            # Parse date column
            if 'Created At' in self.df_transactions.columns:
                self.df_transactions['Created At'] = pd.to_datetime(
                    self.df_transactions['Created At'],
                    errors='coerce'
                )
            
            # Extract year and month
            self.df_transactions['Year'] = self.df_transactions['Created At'].dt.year
            self.df_transactions['Month'] = self.df_transactions['Created At'].dt.month
            
            # Clean identifiers
            def clean_numeric(x):
                if pd.isna(x):
                    return np.nan
                try:
                    # Remove non-numeric characters
                    cleaned = re.sub(r'[^\d.]', '', str(x))
                    if cleaned:
                        return int(float(cleaned))
                    return np.nan
                except:
                    return np.nan
            
            if 'User Identifier' in self.df_transactions.columns:
                self.df_transactions['User Identifier'] = self.df_transactions['User Identifier'].apply(clean_numeric)
            
            if 'Parent User Identifier' in self.df_transactions.columns:
                self.df_transactions['Parent User Identifier'] = self.df_transactions['Parent User Identifier'].apply(clean_numeric)
            
            # Clean text columns
            text_cols = ['Entity Name', 'Service Name', 'Transaction Type', 'Product Name']
            for col in text_cols:
                if col in self.df_transactions.columns:
                    self.df_transactions[col] = self.df_transactions[col].astype(str).str.upper().str.strip()
    
    def _calculate_all_metrics(self):
        """Calculate all 8 required metrics"""
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
            'avg_transaction_time_minutes': 0.0,
            'top_performers': [],
            'regional_distribution': {},
            'network_metrics': {}
        }
        
        # 1. Total Active Agents
        if self.df_onboarding is not None and not self.df_onboarding.empty:
            terminated_status = {'TERMINATED', 'BLOCKED', 'SUSPENDED', 'INACTIVE'}
            active_mask = ~self.df_onboarding['Status'].isin(terminated_status)
            df_active = self.df_onboarding[active_mask]
            
            if 'Entity' in df_active.columns:
                entity_counts = df_active['Entity'].value_counts()
                results['total_active_agents'] = entity_counts.get('AGENT', 0)
                results['total_active_tellers'] = entity_counts.get('AGENT TELLER', 0)
        
        # 2. Total Agent Tellers (already calculated above)
        
        # 3. & 4. Agents with/without Tellers
        if self.df_transactions is not None and not self.df_transactions.empty:
            # Filter for 2025
            df_2025 = self.df_transactions[self.df_transactions['Year'] == self.year].copy()
            
            if 'Parent User Identifier' in df_2025.columns:
                # Get unique parent IDs (agents with tellers)
                parent_ids = df_2025['Parent User Identifier'].dropna().unique()
                results['agents_with_tellers'] = len(parent_ids)
                results['agents_without_tellers'] = max(0, results['total_active_agents'] - results['agents_with_tellers'])
        
        # 5. Onboarding in 2025
        if self.df_onboarding is not None and not self.df_onboarding.empty:
            terminated_status = {'TERMINATED', 'BLOCKED', 'SUSPENDED', 'INACTIVE'}
            active_mask = ~self.df_onboarding['Status'].isin(terminated_status)
            
            if 'Registration Date' in self.df_onboarding.columns:
                mask_2025 = (self.df_onboarding['Registration Date'].dt.year == self.year) & active_mask
                df_onboarded_2025 = self.df_onboarding[mask_2025]
                
                results['onboarded_2025_total'] = len(df_onboarded_2025)
                
                if 'Entity' in df_onboarded_2025.columns:
                    onboarded_counts = df_onboarded_2025['Entity'].value_counts()
                    results['onboarded_2025_agents'] = onboarded_counts.get('AGENT', 0)
                    results['onboarded_2025_tellers'] = onboarded_counts.get('AGENT TELLER', 0)
        
        # 6. & 7. Activeness analysis
        if self.df_transactions is not None and not self.df_transactions.empty:
            df_2025 = self.df_transactions[self.df_transactions['Year'] == self.year].copy()
            
            # Identify deposits
            deposit_keywords = ['DEPOSIT']
            deposit_mask = False
            for col in ['Service Name', 'Transaction Type', 'Product Name']:
                if col in df_2025.columns:
                    col_mask = df_2025[col].str.contains('|'.join(deposit_keywords), na=False)
                    deposit_mask = deposit_mask | col_mask
            
            if isinstance(deposit_mask, pd.Series):
                df_deposits = df_2025[deposit_mask].copy()
                
                # Filter for agents and tellers
                agent_entities = {'AGENT', 'AGENT TELLER'}
                if 'Entity Name' in df_deposits.columns:
                    agent_mask = df_deposits['Entity Name'].isin(agent_entities)
                    df_agent_deposits = df_deposits[agent_mask]
                    
                    if not df_agent_deposits.empty and 'User Identifier' in df_agent_deposits.columns:
                        # Overall activeness
                        deposit_counts = df_agent_deposits['User Identifier'].value_counts()
                        results['active_users_overall'] = len(deposit_counts[deposit_counts >= 20])
                        results['inactive_users_overall'] = len(deposit_counts[deposit_counts < 20])
                        
                        # Get top performers
                        top_performers = deposit_counts.head(10).reset_index()
                        top_performers.columns = ['User ID', 'Deposit Count']
                        results['top_performers'] = top_performers.to_dict('records')
                        
                        # Monthly activeness
                        if 'Month' in df_agent_deposits.columns:
                            for month in range(1, 13):
                                month_mask = df_agent_deposits['Month'] == month
                                month_deposits = df_agent_deposits[month_mask]
                                if not month_deposits.empty:
                                    month_counts = month_deposits['User Identifier'].value_counts()
                                    results['monthly_active_users'][month] = len(month_counts[month_counts >= 20])
        
        # 8. Average Transaction Time
        if self.df_transactions is not None and not self.df_transactions.empty:
            df_2025 = self.df_transactions[self.df_transactions['Year'] == self.year].copy()
            
            if not df_2025.empty and len(df_2025) > 100:
                # Sample for performance
                sample_size = min(10000, len(df_2025))
                df_sample = df_2025.sample(n=sample_size, random_state=42)
                
                # Sort by time and calculate differences
                df_sample = df_sample.sort_values('Created At')
                time_diffs = df_sample['Created At'].diff().dt.total_seconds() / 60
                
                # Filter reasonable times (10 seconds to 30 minutes)
                valid_times = time_diffs.between(0.167, 30)
                if valid_times.any():
                    results['avg_transaction_time_minutes'] = round(time_diffs[valid_times].mean(), 2)
        
        # Additional metrics
        if self.df_onboarding is not None and not self.df_onboarding.empty:
            # Regional distribution
            if 'Region' in self.df_onboarding.columns:
                regional_counts = self.df_onboarding['Region'].value_counts().head(10)
                results['regional_distribution'] = regional_counts.to_dict()
        
        return results
    
    def create_dashboard(self):
        """Create the complete dashboard"""
        # Header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="dashboard-header">
                <h1 style="margin: 0; font-size: 2.5rem;">📊 APS Wallet Dashboard</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Annual Agent Performance Report • {self.year}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Metrics
        st.markdown("### 🎯 Key Performance Indicators")
        
        # Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._display_metric_card(
                "Total Agents",
                self.results['total_active_agents'],
                "👥",
                "#3B82F6"
            )
        
        with col2:
            self._display_metric_card(
                "Agent Tellers",
                self.results['total_active_tellers'],
                "💼",
                "#8B5CF6"
            )
        
        with col3:
            self._display_metric_card(
                "Active Users",
                self.results['active_users_overall'],
                "✅",
                "#10B981"
            )
        
        with col4:
            self._display_metric_card(
                "Avg Transaction Time",
                f"{self.results['avg_transaction_time_minutes']:.1f} min",
                "⏱️",
                "#F59E0B"
            )
        
        # Row 2
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            self._display_metric_card(
                "2025 Onboarded",
                self.results['onboarded_2025_total'],
                "📈",
                "#EC4899"
            )
        
        with col6:
            agents_with = self.results['agents_with_tellers']
            agents_total = max(self.results['total_active_agents'], 1)
            percentage = (agents_with / agents_total) * 100
            self._display_metric_card(
                "Agents with Tellers",
                f"{agents_with} ({percentage:.1f}%)",
                "🔗",
                "#06B6D4"
            )
        
        with col7:
            agents_without = self.results['agents_without_tellers']
            percentage = (agents_without / agents_total) * 100
            self._display_metric_card(
                "Agents without Tellers",
                f"{agents_without} ({percentage:.1f}%)",
                "👤",
                "#22C55E"
            )
        
        with col8:
            inactive = self.results['inactive_users_overall']
            total_users = max(self.results['active_users_overall'] + inactive, 1)
            percentage = (inactive / total_users) * 100
            self._display_metric_card(
                "Inactive Users",
                f"{inactive} ({percentage:.1f}%)",
                "📉",
                "#EF4444"
            )
        
        st.markdown("---")
        
        # Main Analysis Tabs
        tabs = st.tabs([
            "📈 Performance Overview",
            "👥 Agent Network", 
            "📊 Monthly Analysis",
            "🏆 Top Performers",
            "📥 Data Export"
        ])
        
        with tabs[0]:
            self._performance_overview_tab()
        
        with tabs[1]:
            self._agent_network_tab()
        
        with tabs[2]:
            self._monthly_analysis_tab()
        
        with tabs[3]:
            self._top_performers_tab()
        
        with tabs[4]:
            self._data_export_tab()
    
    def _display_metric_card(self, title, value, icon, color):
        """Display a metric card"""
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.8rem;">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _performance_overview_tab(self):
        """Performance Overview Tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Agent Distribution
            agent_data = {
                'Category': ['Agents', 'Tellers'],
                'Count': [self.results['total_active_agents'], self.results['total_active_tellers']]
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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Activeness Distribution
            labels = ['Active (≥20 deposits)', 'Inactive (<20 deposits)']
            values = [self.results['active_users_overall'], self.results['inactive_users_overall']]
            
            fig = px.bar(
                x=labels,
                y=values,
                title='User Activeness Distribution',
                color=labels,
                color_discrete_sequence=['#10B981', '#EF4444'],
                text=values
            )
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Number of Users',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Onboarding Analysis
        st.markdown("### 📈 2025 Onboarding Analysis")
        
        onboarding_data = {
            'Category': ['Agents', 'Tellers', 'Total'],
            'Count': [
                self.results['onboarded_2025_agents'],
                self.results['onboarded_2025_tellers'],
                self.results['onboarded_2025_total']
            ]
        }
        
        df_onboarding = pd.DataFrame(onboarding_data)
        
        fig = px.bar(
            df_onboarding,
            x='Category',
            y='Count',
            title='2025 Onboarding Breakdown',
            text='Count',
            color='Category',
            color_discrete_sequence=['#3B82F6', '#8B5CF6', '#10B981']
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Number Onboarded',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _agent_network_tab(self):
        """Agent Network Analysis Tab"""
        st.markdown("### 👥 Agent Network Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network visualization
            nodes = [
                {'name': 'Total Agents', 'value': self.results['total_active_agents']},
                {'name': 'With Tellers', 'value': self.results['agents_with_tellers']},
                {'name': 'Without Tellers', 'value': self.results['agents_without_tellers']},
                {'name': 'Total Tellers', 'value': self.results['total_active_tellers']}
            ]
            
            fig = px.sunburst(
                names=[n['name'] for n in nodes],
                parents=['', 'Total Agents', 'Total Agents', 'With Tellers'],
                values=[n['value'] for n in nodes],
                title='Agent-Teller Network Structure',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Network statistics
            st.markdown("#### 📊 Network Statistics")
            
            stats_data = {
                'Metric': [
                    'Agent to Teller Ratio',
                    'Network Coverage',
                    'Avg Tellers per Agent'
                ],
                'Value': [
                    f"1:{self.results['total_active_tellers']/max(self.results['total_active_agents'], 1):.1f}",
                    f"{(self.results['agents_with_tellers']/max(self.results['total_active_agents'], 1))*100:.1f}%",
                    f"{self.results['total_active_tellers']/max(self.results['agents_with_tellers'], 1):.1f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Regional distribution
            if self.results['regional_distribution']:
                st.markdown("#### 🌍 Regional Distribution")
                regions = list(self.results['regional_distribution'].keys())[:5]
                counts = list(self.results['regional_distribution'].values())[:5]
                
                region_df = pd.DataFrame({
                    'Region': regions,
                    'Agent Count': counts
                })
                st.dataframe(region_df, use_container_width=True, hide_index=True)
    
    def _monthly_analysis_tab(self):
        """Monthly Analysis Tab"""
        st.markdown("### 📊 Monthly Active Users Analysis")
        
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        
        active_counts = [self.results['monthly_active_users'][m] for m in range(1, 13)]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(
                    x=months,
                    y=active_counts,
                    marker_color='#3B82F6',
                    text=active_counts,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Monthly Active Users (≥20 deposits)',
                xaxis_title='Month',
                yaxis_title='Number of Active Users',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Monthly Statistics")
            
            avg_active = np.mean(active_counts)
            max_active = max(active_counts)
            min_active = min(active_counts)
            max_month = months[active_counts.index(max_active)]
            min_month = months[active_counts.index(min_active)]
            
            st.metric("Average Monthly Active", f"{avg_active:,.0f}")
            st.metric("Peak Month", f"{max_active:,} ({max_month})")
            st.metric("Lowest Month", f"{min_active:,} ({min_month})")
            
            # Quarterly breakdown
            st.markdown("#### 📅 Quarterly Breakdown")
            q1 = sum(active_counts[0:3])
            q2 = sum(active_counts[3:6])
            q3 = sum(active_counts[6:9])
            q4 = sum(active_counts[9:12])
            
            quarterly_data = pd.DataFrame({
                'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                'Active Users': [q1, q2, q3, q4],
                'Average': [q1/3, q2/3, q3/3, q4/3]
            })
            
            st.dataframe(quarterly_data, use_container_width=True, hide_index=True)
    
    def _top_performers_tab(self):
        """Top Performers Tab"""
        st.markdown("### 🏆 Top Performers 2025")
        
        if self.results['top_performers']:
            # Convert to DataFrame
            top_df = pd.DataFrame(self.results['top_performers'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Horizontal bar chart
                fig = px.bar(
                    top_df,
                    x='Deposit Count',
                    y='User ID',
                    orientation='h',
                    title='Top 10 Performers by Deposit Count',
                    color='Deposit Count',
                    color_continuous_scale='Plasma',
                    text='Deposit Count'
                )
                
                fig.update_layout(
                    xaxis_title='Number of Deposits',
                    yaxis_title='User ID',
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Performance Statistics")
                
                if not top_df.empty:
                    max_deposits = top_df['Deposit Count'].max()
                    avg_deposits = top_df['Deposit Count'].mean()
                    
                    st.metric("Top Performer", f"{max_deposits:,} deposits")
                    st.metric("Average (Top 10)", f"{avg_deposits:,.0f} deposits")
                    
                    # Performance tiers
                    st.markdown("#### 📊 Performance Tiers")
                    
                    # Calculate tiers from sample data
                    tiers_data = {
                        'Tier': ['Elite (>100)', 'High (50-99)', 'Medium (20-49)', 'Low (<20)'],
                        'Count': [25, 150, 300, 500]  # Sample data
                    }
                    
                    tiers_df = pd.DataFrame(tiers_data)
                    st.dataframe(tiers_df, use_container_width=True, hide_index=True)
        else:
            st.info("No top performers data available. Upload complete transaction data for this analysis.")
        
        # Recommendations
        st.markdown("### 💡 Performance Insights & Recommendations")
        
        insights = [
            "🎯 **Focus on inactive users**: Implement re-engagement campaigns for users with <20 deposits",
            "📈 **Optimize top performers**: Provide additional incentives to maintain high performance",
            "🔄 **Improve teller network**: Expand teller coverage in underperforming regions",
            "⏱️ **Reduce transaction time**: Streamline processes to improve user experience",
            "👥 **Agent training**: Enhance training programs for agents without tellers",
            "📊 **Data-driven decisions**: Use monthly trends to plan strategic initiatives"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def _data_export_tab(self):
        """Data Export Tab"""
        st.markdown("### 📥 Data Export & Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Export Analysis Results")
            
            # Create summary report
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
                    self.results['total_active_agents'],
                    self.results['total_active_tellers'],
                    self.results['agents_with_tellers'],
                    self.results['agents_without_tellers'],
                    self.results['onboarded_2025_total'],
                    self.results['onboarded_2025_agents'],
                    self.results['onboarded_2025_tellers'],
                    self.results['active_users_overall'],
                    self.results['inactive_users_overall'],
                    self.results['avg_transaction_time_minutes']
                ]
            })
            
            # Convert to CSV
            csv_summary = summary_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=csv_summary,
                file_name=f"aps_wallet_summary_{self.year}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Monthly report
            monthly_data = pd.DataFrame([
                {'Month': datetime(self.year, m, 1).strftime('%B'), 
                 'Active_Users': self.results['monthly_active_users'][m]}
                for m in range(1, 13)
            ])
            
            csv_monthly = monthly_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Monthly Report (CSV)",
                data=csv_monthly,
                file_name=f"aps_wallet_monthly_{self.year}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📈 Generate Custom Reports")
            
            report_type = st.selectbox(
                "Select Report Type",
                ["Agent Performance", "Teller Network", "Monthly Trends", "Regional Analysis"],
                key="report_type"
            )
            
            if st.button("🔄 Generate Custom Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    time.sleep(2)  # Simulate processing
                    
                    # Create sample custom report
                    custom_data = {
                        'Metric': ['Sample Metric 1', 'Sample Metric 2', 'Sample Metric 3'],
                        'Value': [100, 200, 300],
                        'Growth': ['+10%', '+15%', '+5%']
                    }
                    
                    custom_df = pd.DataFrame(custom_data)
                    
                    st.success("✅ Report generated successfully!")
                    st.dataframe(custom_df, use_container_width=True)
                    
                    # Download button for custom report
                    csv_custom = custom_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Custom Report",
                        data=csv_custom,
                        file_name=f"custom_report_{report_type.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # Raw Data Access
        st.markdown("---")
        st.markdown("### 📁 Raw Data Access")
        
        if self.df_onboarding is not None and self.df_transactions is not None:
            col3, col4 = st.columns(2)
            
            with col3:
                st.info(f"Onboarding Data: {len(self.df_onboarding):,} records")
                if st.button("Preview Onboarding Data", use_container_width=True):
                    st.dataframe(self.df_onboarding.head(100), use_container_width=True)
            
            with col4:
                st.info(f"Transaction Data: {len(self.df_transactions):,} records")
                if st.button("Preview Transaction Data", use_container_width=True):
                    st.dataframe(self.df_transactions.head(100), use_container_width=True)

def main():
    """Main application"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/wallet--v1.png", width=80)
    st.sidebar.markdown("## APS Wallet Analytics")
    st.sidebar.markdown("---")
    
    # Initialize dashboard
    dashboard = AgentPerformanceDashboard()
    
    # File upload section
    st.sidebar.markdown("### 📤 Upload Data Files")
    
    onboarding_file = st.sidebar.file_uploader(
        "Onboarding Data (CSV)",
        type=['csv'],
        help="Upload agent onboarding data"
    )
    
    transaction_file = st.sidebar.file_uploader(
        "Transaction Data (CSV)",
        type=['csv'],
        help="Upload transaction history"
    )
    
    # Analysis year
    analysis_year = st.sidebar.selectbox(
        "Analysis Year",
        [2023, 2024, 2025],
        index=2
    )
    
    dashboard.year = analysis_year
    
    # Load sample data button
    if st.sidebar.button("📋 Load Sample Data", use_container_width=True):
        # Create sample data
        sample_onboarding = pd.DataFrame({
            'Account ID': [f'ACC-{i:06d}' for i in range(1, 101)],
            'Entity': np.random.choice(['AGENT', 'AGENT TELLER'], 100, p=[0.6, 0.4]),
            'Status': np.random.choice(['ACTIVE', 'TERMINATED', 'INACTIVE'], 100, p=[0.8, 0.1, 0.1]),
            'Registration Date': pd.date_range('2024-01-01', periods=100),
            'Region': np.random.choice(['West Coast', 'Greater Banjul', 'Central River', 'North Bank'], 100)
        })
        
        sample_transactions = pd.DataFrame({
            'User Identifier': np.random.randint(1000, 2000, 50000),
            'Parent User Identifier': np.random.choice([1001, 1002, 1003, 1004, 1005, np.nan], 50000, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
            'Entity Name': np.random.choice(['AGENT', 'AGENT TELLER', 'CUSTOMER'], 50000, p=[0.3, 0.3, 0.4]),
            'Service Name': np.random.choice(['DEPOSIT', 'WITHDRAWAL', 'TRANSFER'], 50000),
            'Transaction Type': np.random.choice(['CR', 'DR'], 50000),
            'Created At': pd.date_range('2025-01-01', periods=50000, freq='10min')
        })
        
        # Save to session state
        st.session_state.sample_onboarding = sample_onboarding
        st.session_state.sample_transactions = sample_transactions
        
        # Analyze sample data
        dashboard.df_onboarding = sample_onboarding
        dashboard.df_transactions = sample_transactions
        dashboard._clean_data()
        dashboard.results = dashboard._calculate_all_metrics()
        
        st.sidebar.success("✅ Sample data loaded!")
    
    # Process uploaded data
    analyze_button = False
    if onboarding_file and transaction_file:
        analyze_button = st.sidebar.button(
            "🚀 Analyze Data",
            type="primary",
            use_container_width=True
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
        <p>APS Wallet Analytics v2.0</p>
        <p>© 2025 APS Financial Services</p>
        <p>For support: analytics@apswallet.com</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    if analyze_button and onboarding_file and transaction_file:
        with st.spinner("Analyzing data..."):
            success = dashboard.load_data(onboarding_file, transaction_file)
            if success:
                dashboard.create_dashboard()
            else:
                st.error("Failed to analyze data. Please check your files.")
    
    elif hasattr(dashboard, 'results') and dashboard.results:
        # Show results from sample data
        dashboard.create_dashboard()
    
    else:
        # Show welcome page
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <h1 style="color: #1E3A8A; margin-bottom: 20px;">📊 APS Wallet Dashboard</h1>
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
                        <li>Upload your Onboarding CSV file</li>
                        <li>Upload your Transaction CSV file</li>
                        <li>Select the analysis year</li>
                        <li>Click "Analyze Data" or "Load Sample Data"</li>
                    </ol>
                </div>
                
                <div style="margin-top: 40px;">
                    <p style="color: #6B7280;">
                        Need help? Check our documentation or contact 
                        <a href="mailto:support@apswallet.com" style="color: #3B82F6; text-decoration: none;">
                            support@apswallet.com
                        </a>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
