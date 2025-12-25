"""
Advanced Analytics Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Analytics | APS Wallet", layout="wide")

def main():
    st.title("📊 Advanced Analytics")
    st.markdown("Deep dive into agent performance metrics and trends")
    
    # Check for data
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("Please load data from the main page first.")
        return
    
    results = st.session_state.analysis_results
    
    # Analytics Tabs
    tabs = st.tabs([
        "📈 Performance Trends",
        "🔍 Segmentation",
        "📊 Comparative Analysis",
        "📋 Insights"
    ])
    
    with tabs[0]:
        st.markdown("### 📈 Performance Trends Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend line chart
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Simulate trend data
            trend_data = {
                'Month': months,
                'Active Users': [results['monthly_active_users'].get(m, 0) for m in range(1, 13)],
                'Deposits': np.random.randint(1000, 5000, 12),
                'Growth Rate': np.random.uniform(-5, 20, 12)
            }
            
            df_trend = pd.DataFrame(trend_data)
            
            fig = px.line(
                df_trend,
                x='Month',
                y=['Active Users', 'Deposits'],
                title='Monthly Performance Trends',
                markers=True
            )
            
            fig.update_layout(
                yaxis_title="Count",
                height=400,
                legend_title="Metric"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance forecasting
            st.markdown("#### 🔮 Performance Forecasting")
            
            # Moving average
            active_users = df_trend['Active Users'].values
            ma_3 = pd.Series(active_users).rolling(window=3).mean().values
            ma_6 = pd.Series(active_users).rolling(window=6).mean().values
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=active_users,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#3B82F6', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=months,
                y=ma_3,
                mode='lines',
                name='3-Month MA',
                line=dict(color='#10B981', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=months,
                y=ma_6,
                mode='lines',
                name='6-Month MA',
                line=dict(color='#8B5CF6', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title='Trend Analysis with Moving Averages',
                yaxis_title="Active Users",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 🔍 User Segmentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segmentation by activity level
            segments = {
                'Segment': ['Elite (>100)', 'High (50-99)', 'Medium (20-49)', 'Low (<20)'],
                'Users': [250, 450, 850, 405],
                'Avg Deposits': [156, 72, 32, 8]
            }
            
            df_segments = pd.DataFrame(segments)
            
            fig = px.bar(
                df_segments,
                x='Segment',
                y='Users',
                title='User Segmentation by Activity Level',
                color='Avg Deposits',
                color_continuous_scale='Viridis',
                text='Users'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Segment characteristics
            st.markdown("#### 📊 Segment Characteristics")
            
            segment_stats = pd.DataFrame({
                'Metric': ['Avg Transaction Time', 'Retention Rate', 'Referral Rate', 'Satisfaction'],
                'Elite': ['1.2 min', '95%', '45%', '92%'],
                'High': ['1.8 min', '85%', '32%', '88%'],
                'Medium': ['2.5 min', '75%', '18%', '82%'],
                'Low': ['3.5 min', '55%', '8%', '75%']
            })
            
            st.dataframe(segment_stats, use_container_width=True, hide_index=True)
            
            # Segment recommendations
            st.markdown("#### 💡 Segment Strategies")
            strategies = [
                "**Elite**: Premium services, loyalty rewards",
                "**High**: Cross-selling opportunities",
                "**Medium**: Engagement campaigns, training",
                "**Low**: Re-engagement, incentives"
            ]
            
            for strategy in strategies:
                st.markdown(f"- {strategy}")
    
    with tabs[2]:
        st.markdown("### 📊 Comparative Analysis")
        
        # Year-over-Year Comparison
        years = ['2023', '2024', '2025']
        comparison_data = {
            'Year': years,
            'Active Agents': [1200, 1500, 1798],
            'Agent Tellers': [800, 1100, 1370],
            'Active Users': [1400, 1700, 1954],
            'Growth Rate': [12.5, 15.2, 15.8]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = px.line(
            df_comparison,
            x='Year',
            y=['Active Agents', 'Agent Tellers', 'Active Users'],
            title='Year-over-Year Performance Comparison',
            markers=True
        )
        
        fig.update_layout(
            yaxis_title="Count",
            height=500,
            legend_title="Metric"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Matrix
        st.markdown("#### 📈 Performance Matrix")
        
        matrix_data = pd.DataFrame({
            'Region': ['West Coast', 'Greater Banjul', 'Central River', 'North Bank', 'Lower River'],
            'Agent Count': [450, 380, 320, 280, 220],
            'Growth Rate': [18.5, 15.2, 12.8, 9.5, 7.2],
            'Activity Rate': [85, 82, 78, 75, 70],
            'Efficiency': [92, 88, 85, 82, 78]
        })
        
        st.dataframe(
            matrix_data.style.background_gradient(cmap='RdYlGn', subset=['Growth Rate', 'Activity Rate']),
            use_container_width=True
        )
    
    with tabs[3]:
        st.markdown("### 📋 Data Insights")
        
        # Generate insights
        insights = [
            {
                'title': '📈 Growth Opportunity',
                'description': 'Agent network growth rate of 15.8% indicates strong expansion potential',
                'impact': 'High',
                'recommendation': 'Increase teller recruitment in underperforming regions'
            },
            {
                'title': '⏱️ Efficiency Improvement',
                'description': 'Average transaction time of 2.5 minutes can be optimized',
                'impact': 'Medium',
                'recommendation': 'Implement process automation and training'
            },
            {
                'title': '👥 Network Optimization',
                'description': '52% of agents have tellers, indicating room for network expansion',
                'impact': 'High',
                'recommendation': 'Launch teller recruitment incentives'
            },
            {
                'title': '📊 Retention Focus',
                'description': '82.3% retention rate with 17.7% churn rate',
                'impact': 'Medium',
                'recommendation': 'Implement retention programs for at-risk segments'
            }
        ]
        
        for insight in insights:
            with st.expander(f"{insight['title']} ({insight['impact']} Impact)"):
                st.markdown(f"**Description**: {insight['description']}")
                st.markdown(f"**Recommendation**: {insight['recommendation']}")
        
        # Action Plan
        st.markdown("#### 🎯 Recommended Action Plan")
        
        action_plan = pd.DataFrame({
            'Priority': ['P1', 'P1', 'P2', 'P2', 'P3'],
            'Action': [
                'Launch teller recruitment campaign',
                'Optimize transaction processes',
                'Implement retention program',
                'Enhance agent training',
                'Expand to new regions'
            ],
            'Timeline': ['Q1 2026', 'Q1 2026', 'Q2 2026', 'Q2 2026', 'Q3 2026'],
            'Owner': ['Operations', 'Technology', 'Marketing', 'Training', 'Expansion']
        })
        
        st.dataframe(action_plan, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()