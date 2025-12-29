"""
APS Wallet - Annual Agent Performance Dashboard
Production-safe Streamlit App
Python 3.11 compatible
Optimized for large files (4GB+)
"""

# =========================
# Imports
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path
import time

warnings.filterwarnings("ignore")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="APS Wallet | Agent Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Your Windows File Paths
# =========================
ONBOARDING_PATH = r"C:\Users\lamin\Transaction\Onboarding.csv"
TRANSACTIONS_PATH = r"C:\Users\lamin\Transaction\Transactions.csv"

# =========================
# Utilities
# =========================
def format_number(x):
    return f"{int(x):,}"

def format_currency(x):
    return f"GMD {x:,.2f}"

def format_percentage(x):
    return f"{x:.2f}%"

def format_file_size(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

# =========================
# Optimized Data Loader for Large Files
# =========================
class DataLoader:
    @staticmethod
    def get_file_info(file_path):
        """Get file information including size"""
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            return {
                "exists": True,
                "size": size,
                "size_formatted": format_file_size(size),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path))
            }
        return {"exists": False}
    
    @staticmethod
    def load_csv_optimized(file_path, nrows=None, chunksize=100000, required_cols=None):
        """
        Load large CSV files efficiently
        
        Args:
            file_path: Path to CSV file
            nrows: Maximum rows to read (None for all)
            chunksize: Rows per chunk for processing
            required_cols: List of required columns to load
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            # First, read just the column names to check structure
            sample_df = pd.read_csv(file_path, nrows=0)
            
            # If required_cols specified, only load those
            if required_cols:
                available_cols = [col for col in required_cols if col in sample_df.columns]
                if not available_cols:
                    st.warning(f"None of the required columns found in {file_path}")
                    return None
                usecols = available_cols
            else:
                usecols = None
            
            # Estimate total rows for progress bar
            if nrows is None:
                # Count lines for progress estimation (faster than reading entire file)
                with open(file_path, 'rb') as f:
                    line_count = sum(1 for _ in f)
                total_rows = line_count - 1  # Subtract header
            else:
                total_rows = min(nrows, 1000000)  # Cap for progress display
            
            # Initialize progress
            progress_text = f"Loading {Path(file_path).name}..."
            progress_bar = st.progress(0, text=progress_text)
            
            # Read in chunks with optimized dtypes
            chunks = []
            rows_read = 0
            
            for chunk in pd.read_csv(
                file_path,
                chunksize=chunksize,
                nrows=nrows,
                usecols=usecols,
                low_memory=False,
                dtype={
                    'Amount': 'float32',
                    'Status': 'category',
                    'Entity': 'category',
                    'Service Name': 'category'
                }
            ):
                chunks.append(chunk)
                rows_read += len(chunk)
                
                # Update progress
                if total_rows > 0:
                    progress = min(rows_read / total_rows, 1.0)
                    progress_bar.progress(
                        progress,
                        text=f"{progress_text} {rows_read:,} rows loaded"
                    )
            
            progress_bar.empty()
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                st.info(f"✅ Loaded {rows_read:,} rows from {Path(file_path).name}")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None

# =========================
# Analytics Engine (Optimized)
# =========================
class AnalyticsEngine:
    def analyze(self, onboarding_df, transactions_df, year, progress_callback=None):
        """Optimized analysis for large datasets"""
        results = {}
        
        # Update progress
        if progress_callback:
            progress_callback(10, "Processing dates...")
        
        # Process onboarding data
        if onboarding_df is not None and not onboarding_df.empty:
            # Convert dates efficiently
            onboarding_df["Registration Date"] = pd.to_datetime(
                onboarding_df["Registration Date"], errors="coerce", format='mixed'
            )
            
            # Active agents & tellers
            results["total_active_agents"] = onboarding_df[
                (onboarding_df["Entity"] == "AGENT") &
                (onboarding_df["Status"] == "ACTIVE")
            ].shape[0]
            
            results["total_active_tellers"] = onboarding_df[
                (onboarding_df["Entity"].str.contains("TELLER", case=False, na=False)) &
                (onboarding_df["Status"] == "ACTIVE")
            ].shape[0]
            
            # Onboarded in selected year
            results["onboarded_year"] = onboarding_df[
                onboarding_df["Registration Date"].dt.year == year
            ].shape[0]
        else:
            results["total_active_agents"] = 0
            results["total_active_tellers"] = 0
            results["onboarded_year"] = 0
        
        if progress_callback:
            progress_callback(30, "Filtering transactions by year...")
        
        # Process transactions data
        if transactions_df is not None and not transactions_df.empty:
            # Convert dates efficiently
            transactions_df["Created At"] = pd.to_datetime(
                transactions_df["Created At"], errors="coerce", format='mixed'
            )
            
            # Filter by year (more efficient than comparing dt.year directly)
            transactions_df["Year"] = transactions_df["Created At"].dt.year
            year_tx = transactions_df[transactions_df["Year"] == year].copy()
            
            results["total_transactions"] = year_tx.shape[0]
            results["total_volume"] = year_tx["Amount"].sum() if "Amount" in year_tx.columns else 0
            
            if progress_callback:
                progress_callback(50, "Analyzing services...")
            
            # Service breakdown (sample if too large)
            if len(year_tx) > 1000000:
                st.warning("Large dataset detected, sampling 1M rows for service analysis")
                sample_tx = year_tx.sample(n=1000000, random_state=42)
            else:
                sample_tx = year_tx
            
            if "Service Name" in sample_tx.columns and "Amount" in sample_tx.columns:
                service_summary = (
                    sample_tx.groupby("Service Name")["Amount"]
                    .agg(["count", "sum"])
                    .reset_index()
                )
                service_summary.columns = ["Service Name", "Transaction Count", "Total Amount"]
                results["service_summary"] = service_summary
            else:
                results["service_summary"] = pd.DataFrame()
            
            if progress_callback:
                progress_callback(70, "Calculating monthly trends...")
            
            # Monthly trend
            if not year_tx.empty:
                year_tx["month"] = year_tx["Created At"].dt.month
                monthly = (
                    year_tx.groupby("month")["Amount"]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                monthly.columns = ["month", "Total Amount", "Transaction Count"]
                results["monthly_trend"] = monthly
            else:
                results["monthly_trend"] = pd.DataFrame()
                
        else:
            results["total_transactions"] = 0
            results["total_volume"] = 0
            results["service_summary"] = pd.DataFrame()
            results["monthly_trend"] = pd.DataFrame()
        
        if progress_callback:
            progress_callback(90, "Finalizing results...")
        
        # Calculate additional metrics
        if results["total_transactions"] > 0:
            results["avg_transaction"] = results["total_volume"] / results["total_transactions"]
        else:
            results["avg_transaction"] = 0
        
        if progress_callback:
            progress_callback(100, "Analysis complete!")
        
        return results

# =========================
# UI Components
# =========================
def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            box-shadow:0 4px 14px rgba(0,0,0,0.08);
            text-align:center;">
            <h4 style="color:#1E3A8A;margin-bottom:8px;">{title}</h4>
            <h2 style="margin:0;">{value}</h2>
            <p style="color:#6B7280;margin:0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Sidebar Configuration
# =========================
st.sidebar.image(
    "https://img.icons8.com/color/96/wallet--v1.png",
    width=80
)

st.sidebar.markdown(
    "<h2 style='text-align:center;'>APS Wallet</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Display file information
st.sidebar.subheader("📂 File Information")

onboarding_info = DataLoader.get_file_info(ONBOARDING_PATH)
transactions_info = DataLoader.get_file_info(TRANSACTIONS_PATH)

col1, col2 = st.sidebar.columns(2)

with col1:
    if onboarding_info["exists"]:
        st.success("✓ Onboarding")
        st.caption(f"{onboarding_info['size_formatted']}")
    else:
        st.error("✗ Onboarding")

with col2:
    if transactions_info["exists"]:
        st.success("✓ Transactions")
        st.caption(f"{transactions_info['size_formatted']}")
    else:
        st.error("✗ Transactions")

if onboarding_info["exists"] and transactions_info["exists"]:
    total_size = onboarding_info["size"] + transactions_info["size"]
    st.sidebar.info(f"**Total:** {format_file_size(total_size)}")

st.sidebar.markdown("---")

# Analysis settings
st.sidebar.subheader("⚙️ Analysis Settings")

analysis_year = st.sidebar.selectbox(
    "Analysis Year",
    [2023, 2024, 2025],
    index=2
)

# Sampling options for large files
sample_percentage = st.sidebar.slider(
    "Sample Percentage (for testing)",
    min_value=1,
    max_value=100,
    value=100,
    help="Use lower percentage for faster testing with large files"
)

if sample_percentage < 100:
    st.sidebar.warning(f"⚠️ Using {sample_percentage}% sample for testing")

process_btn = st.sidebar.button("🚀 Process Data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center;font-size:12px;'>APS Wallet Analytics v3.0 | Optimized for Large Files</p>",
    unsafe_allow_html=True
)

# =========================
# Session State
# =========================
if "results" not in st.session_state:
    st.session_state.results = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "analysis_year" not in st.session_state:
    st.session_state.analysis_year = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

# =========================
# Progress Callback
# =========================
def update_progress(percent, message):
    progress_bar.progress(percent / 100, text=message)

# =========================
# Process Data
# =========================
if process_btn:
    start_time = time.time()
    
    # Check if files exist
    if not os.path.exists(ONBOARDING_PATH):
        st.error(f"Onboarding file not found: {ONBOARDING_PATH}")
        st.stop()
    
    if not os.path.exists(TRANSACTIONS_PATH):
        st.error(f"Transactions file not found: {TRANSACTIONS_PATH}")
        st.stop()
    
    # Create progress container
    progress_container = st.empty()
    with progress_container.container():
        st.subheader("🔄 Processing Data...")
        progress_bar = st.progress(0, text="Initializing...")
    
    try:
        # Initialize loader and engine
        loader = DataLoader()
        analytics = AnalyticsEngine()
        
        # Calculate sample size
        if sample_percentage < 100:
            # Estimate rows to sample based on file size
            transactions_size = transactions_info["size"]
            target_size = (sample_percentage / 100) * 10000000  # Target ~10M rows max
            sample_rows = int((target_size / transactions_size) * 50000000)  # Rough estimate
            st.info(f"Sampling approximately {sample_rows:,} rows ({sample_percentage}%)")
        else:
            sample_rows = None
        
        # Load data with progress updates
        update_progress(5, "Loading onboarding data...")
        df_onboarding = loader.load_csv_optimized(
            ONBOARDING_PATH,
            required_cols=['Registration Date', 'Entity', 'Status']
        )
        
        update_progress(20, "Loading transactions data...")
        df_transactions = loader.load_csv_optimized(
            TRANSACTIONS_PATH,
            nrows=sample_rows,
            required_cols=['Created At', 'Amount', 'Service Name']
        )
        
        if df_onboarding is None or df_transactions is None:
            st.error("Failed to load data. Check file formats and paths.")
            st.stop()
        
        # Perform analysis
        st.session_state.results = analytics.analyze(
            df_onboarding, df_transactions, analysis_year, update_progress
        )
        
        # Store additional info
        st.session_state.data_loaded = True
        st.session_state.analysis_year = analysis_year
        st.session_state.df_onboarding = df_onboarding
        st.session_state.df_transactions = df_transactions
        
        # Calculate processing time
        processing_time = time.time() - start_time
        st.session_state.processing_time = processing_time
        
        # Clear progress container
        progress_container.empty()
        
        st.success(f"✅ Analysis completed in {processing_time:.1f} seconds!")
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Error during processing: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

# =========================
# Main Content
# =========================
st.markdown(
    f"""
    <h1 style="text-align:center;">
        APS Wallet Annual Report {analysis_year}
    </h1>
    <p style="text-align:center;color:#6B7280;">
        Last updated: {datetime.now().strftime('%d %B %Y %H:%M:%S')}
    </p>
    """,
    unsafe_allow_html=True
)

if st.session_state.results and st.session_state.data_loaded:
    r = st.session_state.results
    
    # Display processing info
    if st.session_state.processing_time > 0:
        st.caption(f"Processed in {st.session_state.processing_time:.1f} seconds")
    
    if sample_percentage < 100:
        st.warning(f"⚠️ Displaying results from {sample_percentage}% sample data")
    
    # =========================
    # KPIs Section
    # =========================
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Active Agents", format_number(r["total_active_agents"]))
    with col2:
        metric_card("Active Tellers", format_number(r["total_active_tellers"]))
    with col3:
        metric_card(f"Onboarded {analysis_year}", format_number(r["onboarded_year"]))
    with col4:
        metric_card("Total Transactions", format_number(r["total_transactions"]))
    
    # =========================
    # Transaction Metrics
    # =========================
    st.markdown("## 💰 Transaction Analytics")
    
    vol_col1, vol_col2, vol_col3 = st.columns(3)
    
    with vol_col1:
        st.metric(
            "Total Volume",
            format_currency(r["total_volume"]),
            help="Total transaction amount for the year"
        )
    
    with vol_col2:
        st.metric(
            "Average Transaction",
            format_currency(r.get("avg_transaction", 0)),
            help="Average amount per transaction"
        )
    
    with vol_col3:
        if r["total_transactions"] > 0 and r["onboarded_year"] > 0:
            tx_per_agent = r["total_transactions"] / r["onboarded_year"]
            st.metric(
                "Tx per New Agent",
                f"{tx_per_agent:,.0f}",
                help="Average transactions per newly onboarded agent"
            )
        else:
            st.metric("Tx per New Agent", "N/A")
    
    # =========================
    # Service Breakdown
    # =========================
    st.markdown("## 🧾 Service Breakdown")
    
    if not r["service_summary"].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_service_value = px.bar(
                r["service_summary"],
                x="Service Name",
                y="Total Amount",
                color="Service Name",
                title=f"Transaction Value by Service ({analysis_year})",
                text="Total Amount"
            )
            fig_service_value.update_traces(
                texttemplate='GMD %{text:,.0f}',
                textposition='outside'
            )
            fig_service_value.update_layout(
                yaxis_title="Amount (GMD)",
                xaxis_title="Service",
                height=500
            )
            st.plotly_chart(fig_service_value, use_container_width=True)
        
        with col2:
            fig_service_count = px.pie(
                r["service_summary"],
                names="Service Name",
                values="Transaction Count",
                title=f"Transaction Count Distribution ({analysis_year})",
                hole=0.3
            )
            fig_service_count.update_traces(
                textinfo='percent+label',
                textposition='inside'
            )
            st.plotly_chart(fig_service_count, use_container_width=True)
        
        # Service details table
        with st.expander("📋 View Service Details"):
            st.dataframe(
                r["service_summary"].sort_values("Total Amount", ascending=False),
                use_container_width=True
            )
    else:
        st.info("No service breakdown data available")
    
    # =========================
    # Monthly Trend
    # =========================
    st.markdown("## 📅 Monthly Transaction Trend")
    
    if not r["monthly_trend"].empty:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Transaction Volume ({analysis_year})', f'Transaction Count ({analysis_year})'),
            vertical_spacing=0.15
        )
        
        # Add volume line
        fig.add_trace(
            go.Scatter(
                x=r["monthly_trend"]["month"],
                y=r["monthly_trend"]["Total Amount"],
                mode='lines+markers+text',
                name='Volume',
                line=dict(color='#1E3A8A', width=3),
                text=[format_currency(x) for x in r["monthly_trend"]["Total Amount"]],
                textposition="top center"
            ),
            row=1, col=1
        )
        
        # Add count bar
        fig.add_trace(
            go.Bar(
                x=r["monthly_trend"]["month"],
                y=r["monthly_trend"]["Transaction Count"],
                name='Count',
                marker_color='#3B82F6',
                text=r["monthly_trend"]["Transaction Count"],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # Update layout
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(
            ticktext=month_names[:len(r["monthly_trend"])],
            tickvals=r["monthly_trend"]["month"],
            row=1, col=1
        )
        fig.update_xaxes(
            ticktext=month_names[:len(r["monthly_trend"])],
            tickvals=r["monthly_trend"]["month"],
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Amount (GMD)", row=1, col=1)
        fig.update_yaxes(title_text="Transaction Count", row=2, col=1)
        fig.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly details table
        with st.expander("📋 View Monthly Details"):
            display_df = r["monthly_trend"].copy()
            display_df["Month"] = display_df["month"].apply(
                lambda x: month_names[x-1] if x <= len(month_names) else f"Month {x}"
            )
            display_df["Avg per Tx"] = display_df["Total Amount"] / display_df["Transaction Count"]
            display_df = display_df[["Month", "Transaction Count", "Total Amount", "Avg per Tx"]]
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No monthly trend data available")
    
    # =========================
    # Data Export Section
    # =========================
    st.markdown("## 📥 Data Export")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Export service summary
        if not r["service_summary"].empty:
            csv_service = r["service_summary"].to_csv(index=False)
            st.download_button(
                "📊 Download Service Summary",
                data=csv_service,
                file_name=f"service_summary_{analysis_year}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col2:
        # Export monthly trend
        if not r["monthly_trend"].empty:
            csv_monthly = r["monthly_trend"].to_csv(index=False)
            st.download_button(
                "📈 Download Monthly Trend",
                data=csv_monthly,
                file_name=f"monthly_trend_{analysis_year}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col3:
        # Export KPI summary
        report_data = {
            'Metric': [
                'Analysis Year',
                'Active Agents',
                'Active Tellers',
                f'Onboarded {analysis_year}',
                'Total Transactions',
                'Total Volume (GMD)',
                'Average Transaction (GMD)',
                'Processing Time (seconds)',
                'Sample Percentage'
            ],
            'Value': [
                analysis_year,
                r["total_active_agents"],
                r["total_active_tellers"],
                r["onboarded_year"],
                r["total_transactions"],
                r["total_volume"],
                r.get("avg_transaction", 0),
                f"{st.session_state.processing_time:.1f}",
                f"{sample_percentage}%"
            ]
        }
        report_df = pd.DataFrame(report_data)
        csv_report = report_df.to_csv(index=False)
        st.download_button(
            "📋 Download KPI Report",
            data=csv_report,
            file_name=f"kpi_report_{analysis_year}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # =========================
    # Data Preview (Optional - commented for large files)
    # =========================
    if st.checkbox("🔍 Preview Sample Data (First 100 rows)", value=False):
        tab1, tab2 = st.tabs(["Onboarding Data", "Transactions Data"])
        
        with tab1:
            if 'df_onboarding' in st.session_state:
                st.dataframe(st.session_state.df_onboarding.head(100))
                st.caption(f"Total rows: {len(st.session_state.df_onboarding):,}")
        
        with tab2:
            if 'df_transactions' in st.session_state:
                st.dataframe(st.session_state.df_transactions.head(100))
                st.caption(f"Total rows: {len(st.session_state.df_transactions):,}")

else:
    # =========================
    # Welcome/Instructions Section
    # =========================
    st.markdown("## 👋 Welcome to APS Wallet Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        ### 📋 Ready to Analyze Your Data!
        
        **Your configured file paths:**
        - **Onboarding:** `C:\\Users\\lamin\\Transaction\\Onboarding.csv`
        - **Transactions:** `C:\\Users\\lamin\\Transaction\\Transactions.csv`
        
        **To get started:**
        1. ✅ Verify files exist at above paths
        2. 📅 Select analysis year (2023-2025)
        3. ⚙️ Adjust sample percentage if needed for testing
        4. 🚀 Click **Process Data** button
        
        **Features:**
        - Optimized for 4GB+ files
        - Progress tracking during loading
        - Memory-efficient processing
        - Exportable reports
        """)
    
    with col2:
        # File status cards
        st.subheader("📁 File Status")
        
        if onboarding_info["exists"]:
            st.success(f"**Onboarding.csv**\n{onboarding_info['size_formatted']}")
        else:
            st.error("Onboarding.csv not found")
            
        if transactions_info["exists"]:
            st.success(f"**Transactions.csv**\n{transactions_info['size_formatted']}")
        else:
            st.error("Transactions.csv not found")
        
        if onboarding_info["exists"] and transactions_info["exists"]:
            total_size = onboarding_info["size"] + transactions_info["size"]
            st.metric("Total Data Size", format_file_size(total_size))
        
        st.markdown("---")
        st.markdown("**Quick Tip:**")
        st.caption("For faster testing with large files, reduce the sample percentage slider")
    
    # Quick action
    if st.button("🚀 Start Processing Full Data", type="primary", use_container_width=True):
        st.session_state.data_loaded = False
        st.rerun()

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#6B7280;font-size:14px;">
        <p>APS Wallet Dashboard v3.0 • Optimized for Large Files • Windows Paths Configured</p>
        <p>Data Source: C:\\Users\\lamin\\Transaction\\</p>
        <p>For support: analytics@apswallet.com | Last run: {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    unsafe_allow_html=True
)
