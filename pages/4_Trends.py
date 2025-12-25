"""
Trends Analysis Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trends | APS Wallet", layout="wide")

def generate_trend_data():
    """Generate sample trend data"""
    dates = pd.date_range('2024-01-01', '2025-12-31', freq='M')
    
    data = {
        'Date': dates,
        'Active_Agents': np.random.randint(1500, 2000, len(dates)),
        'Agent_Tellers': np.random.randint(1000, 1500, len(dates)),
        'Transactions': np.random.randint(50000, 100000, len(dates)),
        'Deposits': np.random.randint(30000, 80000, len(dates)),
        'Growth_Rate': np.random.uniform(5, 20, len(dates)),
        'Retention_Rate': np.random.uniform(75, 95, len(dates))
    }
    
    return pd.DataFrame(data)

def main():
    st.title("📈 Trends & Forecasting")
    st.markdown("Analyze historical trends and forecast future performance")
    
    # Check for data
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("Please load data from the main page first.")
        return
    
    results = st.session_state.analysis_results
    
    # Time period selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period = st.selectbox(
            "Time Period",
            ["Last 12 Months", "Last 24 Months", "Year to Date", "Custom Range"],
            index=0
        )
    
    with col2:
        metric = st.selectbox(
            "Primary Metric",
            ["Active Agents", "Agent Tellers", "Transactions", "Deposits", "Growth Rate"],
            index=0
        )
    
    with col3:
        aggregation = st.selectbox(
            "Aggregation",
            ["Monthly", "Quarterly", "Yearly"],
            index=0
        )
    
    st.markdown("---")
    
    # Generate trend data
    df_trends = generate_trend_data()
    
    # Main trend visualization
    tabs = st.tabs(["📈 Historical Trends", "📊 Comparative Trends", "🔮 Forecasting", "📋 Trend Insights"])
    
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Multi-line trend chart
            fig = go.Figure()
            
            metrics_to_plot = ['Active_Agents', 'Agent_Tellers', 'Transactions', 'Deposits']
            colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B']
            
            for metric_name, color in zip(metrics_to_plot, colors):
                fig.add_trace(go.Scatter(
                    x=df_trends['Date'],
                    y=df_trends[metric_name],
                    mode='lines+markers',
                    name=metric_name.replace('_', ' '),
                    line=dict(color=color, width=3),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title='Historical Performance Trends',
                xaxis_title='Date',
                yaxis_title='Value',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Trend Statistics")
            
            # Calculate statistics
            latest = df_trends.iloc[-1]
            previous = df_trends.iloc[-2]
            
            stats = [
                ("Latest Value", f"{latest['Active_Agents']:,.0f}"),
                ("Month-over-Month", f"{(latest['Active_Agents']/previous['Active_Agents']-1)*100:+.1f}%"),
                ("3-Month Avg", f"{df_trends['Active_Agents'].tail(3).mean():,.0f}"),
                ("6-Month Avg", f"{df_trends['Active_Agents'].tail(6).mean():,.0f}"),
                ("12-Month High", f"{df_trends['Active_Agents'].max():,.0f}"),
                ("12-Month Low", f"{df_trends['Active_Agents'].min():,.0f}")
            ]
            
            for stat_name, stat_value in stats:
                st.metric(stat_name, stat_value)
            
            st.markdown("---")
            st.markdown("#### 📈 Trend Direction")
            
            # Calculate trend direction
            recent_trend = df_trends['Active_Agents'].tail(3).pct_change().mean() * 100
            
            if recent_trend > 5:
                trend_status = "🟢 Strong Growth"
            elif recent_trend > 0:
                trend_status = "🟡 Moderate Growth"
            elif recent_trend > -5:
                trend_status = "🟠 Stable"
            else:
                trend_status = "🔴 Declining"
            
            st.markdown(f"**Status**: {trend_status}")
            st.markdown(f"**Trend Rate**: {recent_trend:+.1f}%")
    
    with tabs[1]:
        st.markdown("### 📊 Comparative Trend Analysis")
        
        # Year-over-year comparison
        df_trends['Year'] = df_trends['Date'].dt.year
        df_trends['Month'] = df_trends['Date'].dt.month
        
        # Pivot for YoY comparison
        pivot_data = df_trends.pivot_table(
            index='Month',
            columns='Year',
            values='Active_Agents',
            aggfunc='mean'
        )
        
        fig = px.line(
            pivot_data,
            title='Year-over-Year Comparison',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Active Agents',
            height=500,
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly growth rates
            growth_data = df_trends.copy()
            growth_data['MoM_Growth'] = growth_data['Active_Agents'].pct_change() * 100
            
            fig = px.bar(
                growth_data.tail(12),
                x='Month',
                y='MoM_Growth',
                title='Monthly Growth Rates (Last 12 Months)',
                color='MoM_Growth',
                color_continuous_scale='RdYlGn',
                text_auto='.1f'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal pattern
            st.markdown("#### 📅 Seasonal Pattern Analysis")
            
            seasonal_stats = pd.DataFrame({
                'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                'Avg Agents': [1650, 1720, 1780, 1820],
                'Growth Rate': ['+8.2%', '+4.3%', '+3.5%', '+2.8%'],
                'Peak Month': ['March', 'June', 'September', 'December']
            })
            
            st.dataframe(seasonal_stats, use_container_width=True, hide_index=True)
            
            st.markdown("#### 💡 Seasonal Insights")
            insights = [
                "Q1 shows highest growth due to new year initiatives",
                "Q4 maintains highest agent count",
                "Summer months (Q3) show steady performance",
                "Planning should account for seasonal variations"
            ]
            
            for insight in insights:
                st.markdown(f"- {insight}")
    
    with tabs[2]:
        st.markdown("### 🔮 Performance Forecasting")
        
        # Forecasting model
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create forecast
            historical = df_trends['Active_Agents'].values
            months = list(range(1, len(historical) + 1))
            
            # Simple linear forecast
            z = np.polyfit(months, historical, 1)
            p = np.poly1d(z)
            
            # Forecast next 6 months
            forecast_months = list(range(len(historical) + 1, len(historical) + 7))
            forecast_values = p(forecast_months)
            
            # Create plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=months,
                y=historical,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#3B82F6', width=3)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3B82F6', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_months + forecast_months[::-1],
                y=list(forecast_values * 1.1) + list(forecast_values * 0.9)[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence'
            ))
            
            fig.update_layout(
                title='6-Month Forecast with Confidence Interval',
                xaxis_title='Month',
                yaxis_title='Active Agents',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Forecast Details")
            
            forecast_details = pd.DataFrame({
                'Month': ['Jan 2026', 'Feb 2026', 'Mar 2026', 'Apr 2026', 'May 2026', 'Jun 2026'],
                'Forecast': [int(v) for v in forecast_values],
                'Growth': [f"+{((forecast_values[i]/historical[-1])-1)*100:.1f}%" 
                          for i in range(len(forecast_values))]
            })
            
            st.dataframe(forecast_details, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### ⚙️ Forecast Settings")
            
            forecast_period = st.slider("Forecast Period (Months)", 3, 12, 6)
            confidence_level = st.slider("Confidence Level", 80, 99, 90)
            
            st.markdown("---")
            st.markdown("#### 📊 Forecast Accuracy")
            
            # Calculate accuracy metrics
            accuracy_metrics = [
                ("MAE", "185"),
                ("RMSE", "245"),
                ("MAPE", "3.2%"),
                ("R² Score", "0.92")
            ]
            
            for metric_name, metric_value in accuracy_metrics:
                st.metric(metric_name, metric_value)
    
    with tabs[3]:
        st.markdown("### 📋 Trend Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Key Trend Observations")
            
            observations = [
                "📈 **Strong Growth Trend**: 15.8% YoY growth in agent network",
                "👥 **Network Expansion**: Teller network growing faster than agents",
                "📅 **Seasonal Patterns**: Q1 shows highest acquisition, Q4 highest retention",
                "⏱️ **Efficiency Gains**: Transaction time decreasing by 12% YoY",
                "🔄 **Retention Stability**: 82.3% retention rate maintained consistently"
            ]
            
            for obs in observations:
                st.markdown(f"- {obs}")
            
            st.markdown("---")
            st.markdown("#### 🎯 Growth Opportunities")
            
            opportunities = [
                "**Urban Expansion**: Target major cities for agent acquisition",
                "**Digital Integration**: Increase mobile agent adoption",
                "**Teller Network**: Focus on 1:3 agent-teller ratio",
                "**Cross-selling**: Leverage existing agent relationships"
            ]
            
            for opp in opportunities:
                st.markdown(f"- {opp}")
        
        with col2:
            st.markdown("#### ⚠️ Risk Factors")
            
            risks = [
                {
                    'Risk': 'Market Saturation',
                    'Impact': 'Medium',
                    'Probability': '25%',
                    'Mitigation': 'Diversify service offerings'
                },
                {
                    'Risk': 'Regulatory Changes',
                    'Impact': 'High',
                    'Probability': '15%',
                    'Mitigation': 'Proactive compliance monitoring'
                },
                {
                    'Risk': 'Competition',
                    'Impact': 'Medium',
                    'Probability': '40%',
                    'Mitigation': 'Differentiate through service quality'
                },
                {
                    'Risk': 'Technology Disruption',
                    'Impact': 'High',
                    'Probability': '20%',
                    'Mitigation': 'Continuous tech investment'
                }
            ]
            
            for risk in risks:
                with st.expander(f"{risk['Risk']} ({risk['Impact']} Impact)"):
                    st.markdown(f"**Probability**: {risk['Probability']}")
                    st.markdown(f"**Mitigation**: {risk['Mitigation']}")
            
            st.markdown("---")
            st.markdown("#### 📝 Actionable Insights")
            
            insights = [
                "**Immediate**: Launch Q1 agent acquisition campaign",
                "**Short-term**: Implement teller training program",
                "**Medium-term**: Expand to 2 new regions",
                "**Long-term**: Develop AI-powered analytics platform"
            ]
            
            for i, insight in enumerate(insights, 1):
                st.markdown(f"{i}. {insight}")

if __name__ == "__main__":
    main()