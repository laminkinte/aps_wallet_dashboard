"""
Reports and Export Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List

st.set_page_config(page_title="Reports | APS Wallet", layout="wide")

def get_table_download_link(df, filename, link_text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none; color: white;">{link_text}</a>'
    return href

def generate_report_data():
    """Generate sample report data"""
    # Monthly report data
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_data = {
        'Month': months,
        'Active_Agents': np.random.randint(1500, 1800, 12),
        'Agent_Tellers': np.random.randint(1000, 1400, 12),
        'New_Onboarding': np.random.randint(100, 300, 12),
        'Active_Users': np.random.randint(1800, 2200, 12),
        'Total_Deposits': np.random.randint(40000, 80000, 12),
        'Avg_Transaction_Time': np.random.uniform(2.0, 3.5, 12)
    }
    
    df_monthly = pd.DataFrame(monthly_data)
    
    # Regional report data
    regions = ['West Coast', 'Greater Banjul', 'Central River', 'North Bank', 'Lower River', 'Upper River']
    regional_data = {
        'Region': regions,
        'Agent_Count': np.random.randint(200, 500, 6),
        'Teller_Count': np.random.randint(150, 400, 6),
        'Agent_Teller_Ratio': np.random.uniform(0.5, 1.0, 6),
        'Growth_Rate': np.random.uniform(5, 25, 6),
        'Activity_Rate': np.random.uniform(70, 90, 6)
    }
    
    df_regional = pd.DataFrame(regional_data)
    
    # Performance report data
    performance_data = {
        'Agent_ID': [f'AG{i:04d}' for i in range(1, 51)],
        'Region': np.random.choice(regions, 50),
        'Deposits_2024': np.random.randint(50, 300, 50),
        'Deposits_2025': np.random.randint(100, 500, 50),
        'Growth': np.random.uniform(-10, 100, 50),
        'Status': np.random.choice(['Active', 'Inactive', 'New'], 50, p=[0.7, 0.2, 0.1])
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    return df_monthly, df_regional, df_performance

def main():
    st.title("📥 Reports & Export")
    st.markdown("Generate and export comprehensive performance reports")
    
    # Check for data
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("Please load data from the main page first.")
        return
    
    results = st.session_state.analysis_results
    
    # Generate report data
    df_monthly, df_regional, df_performance = generate_report_data()
    
    # Report generation section
    st.markdown("## 📋 Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Configure Report")
        
        report_type = st.selectbox(
            "Report Type",
            [
                "Executive Summary",
                "Monthly Performance Report",
                "Agent Performance Report",
                "Regional Analysis Report",
                "Network Analysis Report",
                "Comprehensive Annual Report"
            ]
        )
        
        date_range = st.date_input(
            "Date Range",
            [datetime(2025, 1, 1), datetime(2025, 12, 31)],
            key="report_date_range"
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        format_type = st.radio("Format", ["CSV", "Excel", "PDF", "HTML"], horizontal=True)
        
        if st.button("🔄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                # Simulate report generation
                st.success(f"✅ {report_type} generated successfully!")
                
                # Show preview
                with st.expander("📄 Report Preview", expanded=True):
                    st.markdown(f"""
                    ### {report_type}
                    **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    **Period**: {date_range[0]} to {date_range[1]}
                    
                    #### Summary
                    - Total Agents: {results.get('total_active_agents', 0):,}
                    - Agent Tellers: {results.get('total_active_tellers', 0):,}
                    - Active Users: {results.get('active_users_overall', 0):,}
                    - Growth Rate: {results.get('agent_growth', 0):.1f}%
                    
                    #### Key Findings
                    1. Strong network growth observed
                    2. Transaction efficiency improved
                    3. Regional performance varies
                    4. Retention rates remain stable
                    """)
    
    with col2:
        st.markdown("### 📥 Export Options")
        
        # Quick export buttons
        st.markdown("#### ⚡ Quick Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Monthly Report", use_container_width=True):
                st.info("Preparing monthly report export...")
        
        with col2:
            if st.button("👥 Agent List", use_container_width=True):
                st.info("Preparing agent list export...")
        
        # Custom data export
        st.markdown("#### 🎯 Custom Data Export")
        
        export_options = st.multiselect(
            "Select data to export",
            [
                "Key Metrics",
                "Agent Performance Data",
                "Transaction Summary",
                "Network Analysis",
                "Regional Distribution",
                "Monthly Trends"
            ],
            default=["Key Metrics", "Agent Performance Data"]
        )
        
        # Download buttons for sample data
        st.markdown("---")
        st.markdown("### 📁 Sample Data Downloads")
        
        # Create sample downloads
        sample_files = [
            ("📈 Monthly Performance", df_monthly, "monthly_performance.csv"),
            ("🌍 Regional Analysis", df_regional, "regional_analysis.csv"),
            ("👥 Agent Performance", df_performance, "agent_performance.csv")
        ]
        
        for label, data, filename in sample_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(label)
            with col2:
                st.download_button(
                    label="⬇️ Download",
                    data=data.to_csv(index=False),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}"
                )
    
    st.markdown("---")
    
    # Report Templates Section
    st.markdown("## 📄 Report Templates")
    
    templates = [
        {
            'name': 'Executive Dashboard',
            'description': 'High-level summary for executives',
            'sections': ['Overview', 'Key Metrics', 'Trends', 'Recommendations'],
            'frequency': 'Monthly'
        },
        {
            'name': 'Operational Report',
            'description': 'Detailed operational metrics',
            'sections': ['Performance', 'Efficiency', 'Network', 'Regional'],
            'frequency': 'Weekly'
        },
        {
            'name': 'Agent Performance',
            'description': 'Individual agent performance',
            'sections': ['Rankings', 'Growth', 'Activity', 'Recommendations'],
            'frequency': 'Monthly'
        },
        {
            'name': 'Regional Analysis',
            'description': 'Regional performance comparison',
            'sections': ['Regional Metrics', 'Growth Rates', 'Opportunities', 'Risks'],
            'frequency': 'Quarterly'
        }
    ]
    
    # Display templates
    cols = st.columns(2)
    
    for i, template in enumerate(templates):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {template['name']}")
                st.markdown(f"**Description**: {template['description']}")
                st.markdown(f"**Frequency**: {template['frequency']}")
                
                # Sections
                with st.expander("Report Sections"):
                    for section in template['sections']:
                        st.markdown(f"- {section}")
                
                # Template actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Use Template", key=f"use_{i}"):
                        st.success(f"Selected {template['name']} template")
                with col2:
                    if st.button("📝 Preview", key=f"preview_{i}"):
                        st.info(f"Previewing {template['name']} template")
    
    st.markdown("---")
    
    # Scheduled Reports Section
    st.markdown("## ⏰ Scheduled Reports")
    
    # Current scheduled reports
    scheduled_reports = pd.DataFrame({
        'Report Name': ['Executive Summary', 'Monthly Performance', 'Agent Rankings', 'Regional Analysis'],
        'Frequency': ['Monthly', 'Weekly', 'Monthly', 'Quarterly'],
        'Recipients': ['5', '10', '25', '8'],
        'Next Run': ['2025-01-31', '2025-01-15', '2025-01-31', '2025-03-31'],
        'Status': ['🟢 Active', '🟢 Active', '🟡 Paused', '🟢 Active']
    })
    
    st.dataframe(scheduled_reports, use_container_width=True, hide_index=True)
    
    # Schedule new report
    with st.expander("➕ Schedule New Report", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_report_name = st.text_input("Report Name")
            new_frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
        
        with col2:
            recipients = st.multiselect(
                "Recipients",
                ["executive@apswallet.com", "operations@apswallet.com", 
                 "regional@apswallet.com", "analytics@apswallet.com"]
            )
            start_date = st.date_input("Start Date")
        
        if st.button("📅 Schedule Report", type="primary"):
            if new_report_name and recipients:
                st.success(f"Scheduled {new_report_name} to run {new_frequency}")
            else:
                st.error("Please fill in all required fields")
    
    st.markdown("---")
    
    # Data Management Section
    st.markdown("## 💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗃️ Backup Data", use_container_width=True):
            st.info("Initiating data backup...")
    
    with col2:
        if st.button("🔄 Refresh Cache", use_container_width=True):
            st.info("Refreshing data cache...")
    
    with col3:
        if st.button("🧹 Cleanup Old Data", use_container_width=True):
            st.info("Cleaning up old data files...")
    
    # Data statistics
    st.markdown("#### 📊 Data Statistics")
    
    data_stats = pd.DataFrame({
        'Dataset': ['Onboarding Data', 'Transaction Data', 'Performance Data', 'Network Data'],
        'Records': ['3,168', '13,146,407', '1,798', '12,500'],
        'Last Updated': ['2025-12-25', '2025-12-25', '2025-12-25', '2025-12-25'],
        'Size': ['2.5 MB', '850 MB', '1.2 MB', '3.8 MB']
    })
    
    st.dataframe(data_stats, use_container_width=True, hide_index=True)
    
    # API Access
    st.markdown("---")
    st.markdown("## 🔌 API Access")
    
    with st.expander("API Configuration"):
        st.markdown("""
        ### REST API Endpoints
        
        ```python
        # Get agent performance data
        GET /api/v1/agents/performance
        
        # Get monthly reports
        GET /api/v1/reports/monthly
        
        # Export data
        POST /api/v1/export
        
        # Real-time metrics
        GET /api/v1/metrics/realtime
        ```
        
        ### API Keys
        - **Development**: `dev_aps_2025_key`
        - **Production**: `prod_aps_2025_secure`
        - **Analytics**: `analytics_aps_2025_data`
        
        ### Rate Limits
        - 100 requests per minute
        - 10,000 requests per day
        - Bulk exports: 1 per hour
        """)
        
        # API key management
        if st.button("🔄 Generate New API Key"):
            new_key = f"aps_api_{np.random.randint(100000, 999999)}"
            st.success(f"New API Key: `{new_key}`")
    
    # Support Information
    st.markdown("---")
    st.markdown("## 🆘 Support & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Documentation
        - [User Guide](https://docs.apswallet.com)
        - [API Documentation](https://api.apswallet.com/docs)
        - [Report Templates](https://templates.apswallet.com)
        - [Best Practices](https://bestpractices.apswallet.com)
        """)
    
    with col2:
        st.markdown("""
        ### 💬 Support Channels
        - **Email**: reports@apswallet.com
        - **Phone**: +220 123 4567
        - **Chat**: Available 9AM-5PM
        - **Emergency**: 24/7 critical support
        
        ### 🐛 Report Issues
        - [Bug Reports](https://bugs.apswallet.com)
        - [Feature Requests](https://features.apswallet.com)
        - [Data Issues](https://data.apswallet.com)
        """)

if __name__ == "__main__":
    main()