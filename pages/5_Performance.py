"""
Performance Rankings Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Performance | APS Wallet", layout="wide")

def generate_performance_data():
    """Generate sample performance data"""
    # Top performers
    top_performers = []
    for i in range(1, 21):
        top_performers.append({
            'Rank': i,
            'Agent_ID': f'AG{i:04d}',
            'Region': np.random.choice(['West Coast', 'Greater Banjul', 'Central River', 'North Bank']),
            'Deposits': np.random.randint(100, 500),
            'Transactions': np.random.randint(500, 2000),
            'Volume_GMD': np.random.randint(500000, 5000000),
            'Growth': np.random.uniform(5, 50),
            'Activity_Score': np.random.randint(70, 100)
        })
    
    df_performers = pd.DataFrame(top_performers)
    
    # Regional performance
    regions = ['West Coast', 'Greater Banjul', 'Central River', 'North Bank', 'Lower River', 'Upper River']
    regional_data = {
        'Region': regions,
        'Avg_Deposits': np.random.randint(150, 350, len(regions)),
        'Avg_Transactions': np.random.randint(800, 1800, len(regions)),
        'Avg_Volume': np.random.randint(1000000, 3000000, len(regions)),
        'Agent_Count': np.random.randint(200, 500, len(regions)),
        'Growth_Rate': np.random.uniform(5, 25, len(regions))
    }
    
    df_regional = pd.DataFrame(regional_data)
    
    return df_performers, df_regional

def main():
    st.title("🏆 Performance Rankings")
    st.markdown("Agent and regional performance leaderboards")
    
    # Check for data
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("Please load data from the main page first.")
        return
    
    results = st.session_state.analysis_results
    
    # Generate performance data
    df_performers, df_regional = generate_performance_data()
    
    # Performance metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_deposits = df_performers['Deposits'].max()
        st.metric("Top Deposits", f"{top_deposits:,}")
    
    with col2:
        top_volume = df_performers['Volume_GMD'].max()
        st.metric("Top Volume", f"GMD {top_volume:,}")
    
    with col3:
        avg_growth = df_performers['Growth'].mean()
        st.metric("Avg Growth", f"{avg_growth:.1f}%")
    
    with col4:
        active_agents = len(df_performers)
        st.metric("Active Agents", f"{active_agents:,}")
    
    st.markdown("---")
    
    # Performance tabs
    tabs = st.tabs(["🏅 Top Performers", "📊 Regional Rankings", "📈 Performance Metrics", "🎯 Awards & Recognition"])
    
    with tabs[0]:
        st.markdown("### 🏅 Top 20 Performers")
        
        # Ranking period
        col1, col2, col3 = st.columns(3)
        
        with col1:
            period = st.selectbox(
                "Ranking Period",
                ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
                key="ranking_period"
            )
        
        with col2:
            metric = st.selectbox(
                "Ranking Metric",
                ["Deposits", "Transaction Volume", "Transactions", "Growth Rate", "Activity Score"],
                key="ranking_metric"
            )
        
        with col3:
            region_filter = st.selectbox(
                "Filter by Region",
                ["All Regions"] + df_performers['Region'].unique().tolist(),
                key="region_filter"
            )
        
        # Filter data
        if region_filter != "All Regions":
            df_filtered = df_performers[df_performers['Region'] == region_filter]
        else:
            df_filtered = df_performers.copy()
        
        # Sort by selected metric
        if metric == "Deposits":
            df_filtered = df_filtered.sort_values('Deposits', ascending=False)
        elif metric == "Transaction Volume":
            df_filtered = df_filtered.sort_values('Volume_GMD', ascending=False)
        elif metric == "Transactions":
            df_filtered = df_filtered.sort_values('Transactions', ascending=False)
        elif metric == "Growth Rate":
            df_filtered = df_filtered.sort_values('Growth', ascending=False)
        else:
            df_filtered = df_filtered.sort_values('Activity_Score', ascending=False)
        
        # Reset ranks
        df_filtered['Rank'] = range(1, len(df_filtered) + 1)
        
        # Display rankings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top performers table
            st.dataframe(
                df_filtered.head(20)[['Rank', 'Agent_ID', 'Region', 'Deposits', 'Transactions', 'Volume_GMD', 'Growth', 'Activity_Score']],
                column_config={
                    'Rank': st.column_config.NumberColumn('Rank', width='small'),
                    'Agent_ID': st.column_config.TextColumn('Agent ID'),
                    'Region': st.column_config.TextColumn('Region'),
                    'Deposits': st.column_config.NumberColumn('Deposits', format='%d'),
                    'Transactions': st.column_config.NumberColumn('Transactions', format='%d'),
                    'Volume_GMD': st.column_config.NumberColumn('Volume (GMD)', format='%d'),
                    'Growth': st.column_config.NumberColumn('Growth %', format='%.1f%%'),
                    'Activity_Score': st.column_config.NumberColumn('Activity Score', format='%d')
                },
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Performance distribution
            st.markdown("#### 📊 Performance Distribution")
            
            # Create performance tiers
            tiers = {
                'Tier': ['Elite', 'High', 'Medium', 'Low'],
                'Count': [5, 10, 25, 60],
                'Threshold': ['Top 5%', 'Top 20%', 'Top 50%', 'All']
            }
            
            df_tiers = pd.DataFrame(tiers)
            
            fig = px.pie(
                df_tiers,
                values='Count',
                names='Tier',
                title='Performance Tier Distribution',
                color='Tier',
                color_discrete_sequence=['#FFD700', '#C0C0C0', '#CD7F32', '#6B7280']
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performer stats
            st.markdown("#### 🏆 Top Performer Stats")
            
            if len(df_filtered) > 0:
                top_agent = df_filtered.iloc[0]
                stats = [
                    ("Agent ID", top_agent['Agent_ID']),
                    ("Region", top_agent['Region']),
                    ("Deposits", f"{top_agent['Deposits']:,}"),
                    ("Volume", f"GMD {top_agent['Volume_GMD']:,}"),
                    ("Growth", f"{top_agent['Growth']:.1f}%")
                ]
                
                for stat_name, stat_value in stats:
                    st.metric(stat_name, stat_value)
    
    with tabs[1]:
        st.markdown("### 📊 Regional Performance Rankings")
        
        # Regional rankings
        df_regional_sorted = df_regional.sort_values('Avg_Volume', ascending=False)
        df_regional_sorted['Rank'] = range(1, len(df_regional_sorted) + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional ranking table
            st.dataframe(
                df_regional_sorted[['Rank', 'Region', 'Agent_Count', 'Avg_Deposits', 'Avg_Volume', 'Growth_Rate']],
                column_config={
                    'Rank': st.column_config.NumberColumn('Rank'),
                    'Region': st.column_config.TextColumn('Region'),
                    'Agent_Count': st.column_config.NumberColumn('Agents', format='%d'),
                    'Avg_Deposits': st.column_config.NumberColumn('Avg Deposits', format='%d'),
                    'Avg_Volume': st.column_config.NumberColumn('Avg Volume', format='%d'),
                    'Growth_Rate': st.column_config.NumberColumn('Growth %', format='%.1f%%')
                },
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Regional comparison chart
            fig = px.bar(
                df_regional_sorted,
                x='Region',
                y='Avg_Volume',
                title='Average Transaction Volume by Region',
                color='Growth_Rate',
                color_continuous_scale='Viridis',
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title='Region',
                yaxis_title='Average Volume (GMD)',
                xaxis_tickangle=-45,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional insights
        st.markdown("#### 🌍 Regional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top Performing Region**")
            top_region = df_regional_sorted.iloc[0]
            st.markdown(f"""
            - **Region**: {top_region['Region']}
            - **Average Volume**: GMD {top_region['Avg_Volume']:,}
            - **Growth Rate**: {top_region['Growth_Rate']:.1f}%
            - **Agent Count**: {top_region['Agent_Count']:,}
            """)
        
        with col2:
            st.markdown("**📈 Most Improved Region**")
            most_improved = df_regional_sorted.loc[df_regional_sorted['Growth_Rate'].idxmax()]
            st.markdown(f"""
            - **Region**: {most_improved['Region']}
            - **Growth Rate**: {most_improved['Growth_Rate']:.1f}%
            - **Current Rank**: {df_regional_sorted[df_regional_sorted['Region'] == most_improved['Region']]['Rank'].values[0]}
            - **Key Driver**: High agent acquisition rate
            """)
    
    with tabs[2]:
        st.markdown("### 📈 Detailed Performance Metrics")
        
        # Performance metrics visualization
        metric_tabs = st.tabs(["📊 Deposit Performance", "💰 Volume Analysis", "📈 Growth Trends", "🎯 Activity Metrics"])
        
        with metric_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Deposit distribution
                fig = px.histogram(
                    df_performers,
                    x='Deposits',
                    nbins=20,
                    title='Deposit Distribution',
                    labels={'Deposits': 'Number of Deposits'},
                    color_discrete_sequence=['#3B82F6']
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Deposits vs Volume
                fig = px.scatter(
                    df_performers,
                    x='Deposits',
                    y='Volume_GMD',
                    color='Region',
                    size='Transactions',
                    title='Deposits vs Transaction Volume',
                    hover_name='Agent_ID',
                    size_max=30
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with metric_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume distribution
                fig = px.box(
                    df_performers,
                    x='Region',
                    y='Volume_GMD',
                    title='Transaction Volume Distribution by Region',
                    color='Region'
                )
                
                fig.update_layout(
                    xaxis_title='Region',
                    yaxis_title='Volume (GMD)',
                    xaxis_tickangle=-45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume trends
                volume_stats = pd.DataFrame({
                    'Metric': ['Average Volume', 'Median Volume', 'Top 10% Volume', 'Bottom 10% Volume'],
                    'Value': [
                        f"GMD {df_performers['Volume_GMD'].mean():,.0f}",
                        f"GMD {df_performers['Volume_GMD'].median():,.0f}",
                        f"GMD {df_performers['Volume_GMD'].quantile(0.9):,.0f}",
                        f"GMD {df_performers['Volume_GMD'].quantile(0.1):,.0f}"
                    ]
                })
                
                st.dataframe(volume_stats, use_container_width=True, hide_index=True)
        
        with metric_tabs[2]:
            # Growth analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Growth distribution
                fig = px.histogram(
                    df_performers,
                    x='Growth',
                    nbins=20,
                    title='Growth Rate Distribution',
                    labels={'Growth': 'Growth Rate (%)'},
                    color_discrete_sequence=['#10B981']
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Growth by region
                growth_by_region = df_performers.groupby('Region')['Growth'].agg(['mean', 'std', 'count']).reset_index()
                
                fig = px.bar(
                    growth_by_region,
                    x='Region',
                    y='mean',
                    error_y='std',
                    title='Average Growth Rate by Region',
                    labels={'mean': 'Average Growth Rate (%)'},
                    color='count',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title='Region',
                    yaxis_title='Average Growth Rate (%)',
                    xaxis_tickangle=-45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 🎯 Awards & Recognition")
        
        # Awards categories
        awards_categories = [
            {
                'name': '🏆 Top Performer Award',
                'criteria': 'Highest transaction volume',
                'winner': 'AG0012',
                'achievement': 'GMD 4,850,000 volume'
            },
            {
                'name': '📈 Growth Champion',
                'criteria': 'Highest growth rate',
                'winner': 'AG0045',
                'achievement': '48.2% growth rate'
            },
            {
                'name': '👥 Network Builder',
                'criteria': 'Most tellers recruited',
                'winner': 'AG0023',
                'achievement': '15 tellers recruited'
            },
            {
                'name': '⭐ Rising Star',
                'criteria': 'Best new performer',
                'winner': 'AG0098',
                'achievement': 'Joined 3 months ago'
            }
        ]
        
        # Display awards
        cols = st.columns(2)
        
        for i, award in enumerate(awards_categories):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {award['name']}")
                    st.markdown(f"**Criteria**: {award['criteria']}")
                    st.markdown(f"**Winner**: **{award['winner']}**")
                    st.markdown(f"**Achievement**: {award['achievement']}")
                    
                    # Award badge
                    st.markdown("""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div style="display: inline-block; padding: 0.5rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px; font-weight: bold;">
                            🏅 Award Winner
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Recognition program
        st.markdown("---")
        st.markdown("### 🎖️ Monthly Recognition Program")
        
        recognition_program = pd.DataFrame({
            'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
            'Top Performer': ['AG0012', 'AG0008', 'AG0012', 'AG0045', 'AG0023', 'AG0012'],
            'Growth Champion': ['AG0045', 'AG0098', 'AG0076', 'AG0098', 'AG0045', 'AG0034'],
            'Network Builder': ['AG0023', 'AG0023', 'AG0015', 'AG0023', 'AG0015', 'AG0023']
        })
        
        st.dataframe(recognition_program, use_container_width=True, hide_index=True)
        
        st.markdown("#### 💡 Recognition Insights")
        
        insights = [
            "**Consistent Performance**: AG0012 has been top performer in 3 of 6 months",
            "**Growth Focus**: New agents showing strong growth potential",
            "**Network Effect**: Top network builders consistently recruit new tellers",
            "**Motivation Impact**: Recognition program correlates with 25% performance improvement"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")

if __name__ == "__main__":
    main()