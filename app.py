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
import glob

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
# File Path Configuration
# =========================
# Default paths based on your system
DEFAULT_PATHS = {
    "onboarding": [
        r"C:\Users\lamin\Transaction\Onboarding.csv",
        r"C:\Users\lamin\Downloads\Onboarding.csv",
        r"C:\Users\lamin\Desktop\Onboarding.csv",
        r"C:\Users\lamin\Documents\Onboarding.csv",
        r"D:\Transaction\Onboarding.csv"
    ],
    "transactions": [
        r"C:\Users\lamin\Transaction\Transactions.csv",
        r"C:\Users\lamin\Downloads\Transactions.csv",
        r"C:\Users\lamin\Desktop\Transactions.csv",
        r"C:\Users\lamin\Documents\Transactions.csv",
        r"D:\Transaction\Transactions.csv"
    ]
}

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
    if bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def find_file_in_system(filename):
    """Search for files in common locations"""
    search_paths = [
        r"C:\Users\lamin\*",
        r"D:\*",
        r"E:\*",
        r"C:\Users\*\Downloads\*",
        r"C:\Users\*\Desktop\*",
        r"C:\Users\*\Documents\*"
    ]
    
    found_files = []
    for path in search_paths:
        search_pattern = os.path.join(os.path.dirname(path), filename)
        try:
            matches = glob.glob(search_pattern, recursive=True)
            found_files.extend(matches)
        except:
            continue
    
    return found_files[:5]  # Return first 5 matches

def validate_file_path(file_path):
    """Check if file exists and is readable"""
    if not file_path or not isinstance(file_path, str):
        return False, "No file path provided"
    
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    try:
        # Try to get file size
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "File is empty"
        return True, f"File found ({format_file_size(size)})"
    except Exception as e:
        return False, f"Error accessing file: {str(e)}"

# =========================
# Optimized Data Loader for Large Files
# =========================
class DataLoader:
    @staticmethod
    def get_file_info(file_path):
        """Get file information including size"""
        if not file_path:
            return {"exists": False, "error": "No path provided"}
        
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                size = os.path.getsize(file_path)
                return {
                    "exists": True,
                    "size": size,
                    "size_formatted": format_file_size(size),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)),
                    "path": file_path
                }
            except Exception as e:
                return {"exists": False, "error": str(e)}
        return {"exists": False, "error": "File not found"}
    
    @staticmethod
    def load_csv_optimized(file_path, nrows=None, chunksize=50000, required_cols=None):
        """
        Load large CSV files efficiently
        """
        try:
            if not file_path or not os.path.exists(file_path):
                return None, "File not found"
            
            # First, read just the header to check structure
            try:
                sample_df = pd.read_csv(file_path, nrows=0)
                if sample_df.empty:
                    return None, "File appears to be empty or malformed"
            except Exception as e:
                return None, f"Error reading file header: {str(e)}"
            
            # If required_cols specified, only load those
            if required_cols:
                available_cols = [col for col in required_cols if col in sample_df.columns]
                if not available_cols:
                    return None, f"Required columns not found in file"
                usecols = available_cols
            else:
                usecols = None
            
            # Initialize progress
            filename = os.path.basename(file_path)
            progress_text = f"Loading {filename}..."
            
            # Read in chunks with optimized dtypes
            chunks = []
            rows_read = 0
            
            # Create a placeholder for progress
            progress_placeholder = st.empty()
            progress_placeholder.text(progress_text)
            
            # Get approximate total rows for very large files
            if nrows:
                total_rows = min(nrows, 10000000)
            else:
                # For very large files, we'll estimate
                total_rows = 10000000  # Default estimate
            
            chunk_iter = pd.read_csv(
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
            )
            
            for chunk in chunk_iter:
                chunks.append(chunk)
                rows_read += len(chunk)
                
                # Update progress every 5 chunks
                if rows_read % (chunksize * 5) == 0:
                    progress_placeholder.text(f"{progress_text} {rows_read:,} rows loaded...")
            
            progress_placeholder.empty()
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                return df, f"Loaded {rows_read:,} rows"
            else:
                return pd.DataFrame(), "No data loaded"
                
        except Exception as e:
            return None, f"Error loading file: {str(e)}"

# =========================
# Analytics Engine (Optimized)
# =========================
class AnalyticsEngine:
    def analyze(self, onboarding_df, transactions_df, year):
        """Optimized analysis for large datasets"""
        results = {}
        
        # Process onboarding data
        if onboarding_df is not None and not onboarding_df.empty:
            # Convert dates efficiently
            date_cols = ['Registration Date', 'registration_date', 'REGISTRATION_DATE']
            onboarding_date_col = None
            
            for col in date_cols:
                if col in onboarding_df.columns:
                    onboarding_date_col = col
                    break
            
            if onboarding_date_col:
                onboarding_df[onboarding_date_col] = pd.to_datetime(
                    onboarding_df[onboarding_date_col], errors="coerce", format='mixed'
                )
            
            # Find entity and status columns
            entity_cols = ['Entity', 'entity', 'ENTITY']
            status_cols = ['Status', 'status', 'STATUS']
            
            entity_col = next((col for col in entity_cols if col in onboarding_df.columns), None)
            status_col = next((col for col in status_cols if col in onboarding_df.columns), None)
            
            if entity_col and status_col:
                # Active agents & tellers
                results["total_active_agents"] = onboarding_df[
                    (onboarding_df[entity_col] == "AGENT") &
                    (onboarding_df[status_col] == "ACTIVE")
                ].shape[0]
                
                results["total_active_tellers"] = onboarding_df[
                    (onboarding_df[entity_col].astype(str).str.contains("TELLER", case=False, na=False)) &
                    (onboarding_df[status_col] == "ACTIVE")
                ].shape[0]
                
                # Onboarded in selected year
                if onboarding_date_col:
                    results["onboarded_year"] = onboarding_df[
                        onboarding_df[onboarding_date_col].dt.year == year
                    ].shape[0]
                else:
                    results["onboarded_year"] = 0
            else:
                results["total_active_agents"] = 0
                results["total_active_tellers"] = 0
                results["onboarded_year"] = 0
        else:
            results["total_active_agents"] = 0
            results["total_active_tellers"] = 0
            results["onboarded_year"] = 0
        
        # Process transactions data
        if transactions_df is not None and not transactions_df.empty:
            # Convert dates efficiently
            tx_date_cols = ['Created At', 'created_at', 'CREATED_AT', 'Transaction Date', 'transaction_date']
            tx_date_col = None
            
            for col in tx_date_cols:
                if col in transactions_df.columns:
                    tx_date_col = col
                    break
            
            if tx_date_col:
                transactions_df[tx_date_col] = pd.to_datetime(
                    transactions_df[tx_date_col], errors="coerce", format='mixed'
                )
            
            # Find amount and service columns
            amount_cols = ['Amount', 'amount', 'AMOUNT']
            service_cols = ['Service Name', 'service_name', 'SERVICE_NAME', 'Service', 'service']
            
            amount_col = next((col for col in amount_cols if col in transactions_df.columns), None)
            service_col = next((col for col in service_cols if col in transactions_df.columns), None)
            
            if tx_date_col and amount_col:
                # Filter by year
                transactions_df["Year"] = transactions_df[tx_date_col].dt.year
                year_tx = transactions_df[transactions_df["Year"] == year].copy()
                
                results["total_transactions"] = year_tx.shape[0]
                results["total_volume"] = year_tx[amount_col].sum()
                
                # Service breakdown
                if service_col and not year_tx.empty:
                    # Sample if too large
                    if len(year_tx) > 1000000:
                        sample_size = min(1000000, len(year_tx))
                        sample_tx = year_tx.sample(n=sample_size, random_state=42)
                    else:
                        sample_tx = year_tx
                    
                    service_summary = (
                        sample_tx.groupby(service_col)[amount_col]
                        .agg(["count", "sum"])
                        .reset_index()
                    )
                    service_summary.columns = ["Service Name", "Transaction Count", "Total Amount"]
                    results["service_summary"] = service_summary
                else:
                    results["service_summary"] = pd.DataFrame()
                
                # Monthly trend
                if not year_tx.empty:
                    year_tx["month"] = year_tx[tx_date_col].dt.month
                    monthly = (
                        year_tx.groupby("month")[amount_col]
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
        else:
            results["total_transactions"] = 0
            results["total_volume"] = 0
            results["service_summary"] = pd.DataFrame()
            results["monthly_trend"] = pd.DataFrame()
        
        # Calculate additional metrics
        if results["total_transactions"] > 0:
            results["avg_transaction"] = results["total_volume"] / results["total_transactions"]
        else:
            results["avg_transaction"] = 0
        
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
# Initialize Session State
# =========================
if "onboarding_path" not in st.session_state:
    # Try to find onboarding file
    found_files = find_file_in_system("Onboarding.csv")
    if found_files:
        st.session_state.onboarding_path = found_files[0]
    else:
        st.session_state.onboarding_path = DEFAULT_PATHS["onboarding"][0]

if "transactions_path" not in st.session_state:
    # Try to find transactions file
    found_files = find_file_in_system("Transactions.csv")
    if found_files:
        st.session_state.transactions_path = found_files[0]
    else:
        st.session_state.transactions_path = DEFAULT_PATHS["transactions"][0]

if "results" not in st.session_state:
    st.session_state.results = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

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

# File Path Configuration
st.sidebar.subheader("📂 File Configuration")

# Onboarding file path
onboarding_path = st.sidebar.text_input(
    "Onboarding CSV Path",
    value=st.session_state.onboarding_path,
    help="Enter full path to Onboarding.csv file"
)

# Transactions file path
transactions_path = st.sidebar.text_input(
    "Transactions CSV Path",
    value=st.session_state.transactions_path,
    help="Enter full path to Transactions.csv file"
)

# File search buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔍 Find Onboarding", use_container_width=True):
        found_files = find_file_in_system("Onboarding.csv")
        if found_files:
            st.session_state.onboarding_path = found_files[0]
            st.rerun()
        else:
            st.sidebar.warning("No Onboarding.csv files found")

with col2:
    if st.button("🔍 Find Transactions", use_container_width=True):
        found_files = find_file_in_system("Transactions.csv")
        if found_files:
            st.session_state.transactions_path = found_files[0]
            st.rerun()
        else:
            st.sidebar.warning("No Transactions.csv files found")

# Validate file paths
st.sidebar.markdown("---")
st.sidebar.subheader("✅ File Validation")

onboarding_valid, onboarding_msg = validate_file_path(onboarding_path)
transactions_valid, transactions_msg = validate_file_path(transactions_path)

if onboarding_valid:
    st.sidebar.success(f"Onboarding: {onboarding_msg}")
else:
    st.sidebar.error(f"Onboarding: {onboarding_msg}")

if transactions_valid:
    st.sidebar.success(f"Transactions: {transactions_msg}")
else:
    st.sidebar.error(f"Transactions: {transactions_msg}")

st.sidebar.markdown("---")

# Analysis settings
st.sidebar.subheader("⚙️ Analysis Settings")

analysis_year = st.sidebar.selectbox(
    "Analysis Year",
    [2023, 2024, 2025],
    index=2
)

# Sampling options
sample_percentage = st.sidebar.slider(
    "Sample Percentage",
    min_value=1,
    max_value=100,
    value=10,
    help="Use lower percentage for faster testing with large files"
)

st.sidebar.markdown(f"**Note:** Using {sample_percentage}% sample for testing")

process_btn = st.sidebar.button("🚀 Process Data", type="primary", use_container_width=True, 
                               disabled=not (onboarding_valid and transactions_valid))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center;font-size:12px;'>APS Wallet Analytics v3.1 • File Finder Enabled</p>",
    unsafe_allow_html=True
)

# =========================
# Main Content - Welcome Screen
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

if not onboarding_valid or not transactions_valid:
    # Show troubleshooting guide
    st.markdown("## 🔍 File Not Found - Troubleshooting Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("### Issue Detected")
        st.write("The application cannot find your CSV files. This could be due to:")
        st.write("1. **Incorrect file path**")
        st.write("2. **Files in different location**")
        st.write("3. **File permission issues**")
        st.write("4. **Files have different names**")
        
        st.markdown("### 🔧 Quick Fixes:")
        
        if st.button("🔄 Use Sample Data for Demo", type="secondary"):
            # Generate sample data for demo
            st.session_state.use_sample_data = True
            st.rerun()
    
    with col2:
        st.info("### 📋 How to Find Your Files")
        st.write("**Option 1:** Use the 🔍 Find buttons in sidebar")
        st.write("**Option 2:** Manually enter correct paths:")
        
        st.code("""
        Common locations to check:
        
        C:\\Users\\lamin\\Downloads\\
        C:\\Users\\lamin\\Desktop\\
        C:\\Users\\lamin\\Documents\\
        D:\\ (if you have another drive)
        
        Right-click file → Properties → Copy Location
        """)
        
        st.write("**Option 3:** Check file names are exactly:")
        st.write("- `Onboarding.csv` (case may vary)")
        st.write("- `Transactions.csv` (case may vary)")
    
    # Show file search results
    with st.expander("🔎 Search for CSV files on your system"):
        st.write("Searching for CSV files...")
        
        # Search for CSV files
        csv_files = []
        search_locations = [
            r"C:\Users\lamin\**\*.csv",
            r"D:\**\*.csv",
            r"C:\Users\*\Downloads\**\*.csv",
            r"C:\Users\*\Desktop\**\*.csv"
        ]
        
        for location in search_locations:
            try:
                files = glob.glob(location, recursive=True)
                csv_files.extend(files)
            except:
                continue
        
        # Filter for likely APS files
        aps_files = []
        other_csv = []
        
        for file in csv_files[:50]:  # Limit to first 50
            filename = os.path.basename(file).lower()
            if 'onboarding' in filename or 'transaction' in filename or 'aps' in filename:
                aps_files.append(file)
            else:
                other_csv.append(file)
        
        if aps_files:
            st.success("Found possible APS files:")
            for file in aps_files[:10]:  # Show first 10
                size = format_file_size(os.path.getsize(file)) if os.path.exists(file) else "Unknown"
                if st.button(f"📁 {os.path.basename(file)} ({size})", key=file):
                    if 'onboarding' in file.lower():
                        st.session_state.onboarding_path = file
                    else:
                        st.session_state.transactions_path = file
                    st.rerun()
        
        if other_csv:
            with st.expander("See all CSV files found"):
                for file in other_csv[:20]:
                    st.text(f"• {file}")

# =========================
# Process Data
# =========================
if process_btn:
    if not onboarding_valid or not transactions_valid:
        st.error("Please fix file paths before processing")
        st.stop()
    
    start_time = time.time()
    
    # Create processing container
    with st.spinner(f"Processing {sample_percentage}% of data..."):
        try:
            # Initialize loader and engine
            loader = DataLoader()
            analytics = AnalyticsEngine()
            
            # Calculate sample rows
            transactions_info = loader.get_file_info(transactions_path)
            if transactions_info["exists"]:
                # Estimate rows based on file size (rough estimate: 1MB ≈ 10,000 rows)
                estimated_rows = int((transactions_info["size"] / (1024 * 1024)) * 10000)
                sample_rows = int((sample_percentage / 100) * estimated_rows)
                sample_rows = min(sample_rows, 5000000)  # Cap at 5M rows
            else:
                sample_rows = 100000
            
            # Load data
            st.info(f"Loading data (sampling {sample_rows:,} rows)...")
            
            df_onboarding, onboarding_msg = loader.load_csv_optimized(
                onboarding_path,
                required_cols=['Registration Date', 'registration_date', 'Entity', 'entity', 'Status', 'status']
            )
            
            if df_onboarding is not None:
                st.success(f"Onboarding: {onboarding_msg}")
            else:
                st.error(f"Failed to load onboarding: {onboarding_msg}")
                st.stop()
            
            df_transactions, transactions_msg = loader.load_csv_optimized(
                transactions_path,
                nrows=sample_rows,
                required_cols=['Created At', 'created_at', 'Amount', 'amount', 'Service Name', 'service_name']
            )
            
            if df_transactions is not None:
                st.success(f"Transactions: {transactions_msg}")
            else:
                st.error(f"Failed to load transactions: {transactions_msg}")
                st.stop()
            
            # Perform analysis
            st.info("Analyzing data...")
            st.session_state.results = analytics.analyze(
                df_onboarding, df_transactions, analysis_year
            )
            
            # Store data
            st.session_state.data_loaded = True
            st.session_state.df_onboarding = df_onboarding
            st.session_state.df_transactions = df_transactions
            st.session_state.processing_time = time.time() - start_time
            st.session_state.sample_percentage = sample_percentage
            
            st.success(f"✅ Analysis completed in {st.session_state.processing_time:.1f} seconds!")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("Try reducing the sample percentage or checking file formats")

# =========================
# Display Results
# =========================
if st.session_state.results and st.session_state.data_loaded:
    r = st.session_state.results
    
    # Show sample warning
    if st.session_state.sample_percentage < 100:
        st.warning(f"⚠️ Showing results from {st.session_state.sample_percentage}% sample data")
    
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
        st.metric("Total Volume", format_currency(r["total_volume"]))
    
    with vol_col2:
        st.metric("Average Transaction", format_currency(r.get("avg_transaction", 0)))
    
    with vol_col3:
        st.metric("Sample Size", f"{st.session_state.sample_percentage}%")
    
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
                title=f"Transaction Value by Service ({analysis_year})"
            )
            st.plotly_chart(fig_service_value, use_container_width=True)
        
        with col2:
            fig_service_count = px.pie(
                r["service_summary"],
                names="Service Name",
                values="Transaction Count",
                title=f"Transaction Count Distribution ({analysis_year})"
            )
            st.plotly_chart(fig_service_count, use_container_width=True)
    else:
        st.info("No service breakdown data available")
    
    # =========================
    # Monthly Trend
    # =========================
    st.markdown("## 📅 Monthly Transaction Trend")
    
    if not r["monthly_trend"].empty:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Transaction Volume ({analysis_year})', f'Transaction Count ({analysis_year})')
        )
        
        fig.add_trace(
            go.Scatter(
                x=r["monthly_trend"]["month"],
                y=r["monthly_trend"]["Total Amount"],
                mode='lines+markers',
                name='Volume'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=r["monthly_trend"]["month"],
                y=r["monthly_trend"]["Transaction Count"],
                name='Count'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly trend data available")
    
    # =========================
    # Data Export
    # =========================
    st.markdown("## 📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not r["service_summary"].empty:
            csv_service = r["service_summary"].to_csv(index=False)
            st.download_button(
                "Download Service Summary",
                data=csv_service,
                file_name=f"service_summary_{analysis_year}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not r["monthly_trend"].empty:
            csv_monthly = r["monthly_trend"].to_csv(index=False)
            st.download_button(
                "Download Monthly Trend",
                data=csv_monthly,
                file_name=f"monthly_trend_{analysis_year}.csv",
                mime="text/csv"
            )

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center;color:#6B7280;font-size:14px;">
        <p>APS Wallet Dashboard v3.1 • Windows File Finder Enabled</p>
        <p>Current paths: {onboarding_path} | {transactions_path}</p>
        <p>For support: analytics@apswallet.com</p>
    </div>
    """,
    unsafe_allow_html=True
)
