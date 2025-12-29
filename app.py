"""APS Wallet - Annual Agent Performance Dashboard
Production-safe Streamlit App
Python 3.11 compatible
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
# Utilities
# =========================
def format_number(x):
    return f"{int(x):,}"

def format_currency(x):
    return f"GMD {x:,.2f}"

def format_percentage(x):
    return f"{x:.2f}%"

# =========================
# Data Loader
# =========================
class DataLoader:
    @staticmethod
    def load_csv(file, nrows=None):
        return pd.read_csv(file, low_memory=False, nrows=nrows)

# =========================
# Analytics Engine
# =========================
class AnalyticsEngine:
    def analyze(self, onboarding_df, transactions_df, year):
        results = {}

        onboarding_df["Registration Date"] = pd.to_datetime(
            onboarding_df["Registration Date"], errors="coerce"
        )

        transactions_df["Created At"] = pd.to_datetime(
            transactions_df["Created At"], errors="coerce"
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

        # Transactions summary
        year_tx = transactions_df[
            transactions_df["Created At"].dt.year == year
        ]

        results["total_transactions"] = year_tx.shape[0]
        results["total_volume"] = year_tx["Amount"].sum()

        # Service breakdown
        service_summary = (
            year_tx.groupby("Service Name")["Amount"]
            .agg(["count", "sum"])
            .reset_index()
        )

        results["service_summary"] = service_summary

        # Monthly trend
        year_tx["month"] = year_tx["Created At"].dt.month
        monthly = (
            year_tx.groupby("month")["Amount"]
            .sum()
            .reset_index()
        )

        results["monthly_trend"] = monthly

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
# Sidebar
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

onboarding_file = st.sidebar.file_uploader(
    "Upload Onboarding CSV",
    type=["csv"]
)

transactions_file = st.sidebar.file_uploader(
    "Upload Transactions CSV",
    type=["csv"]
)

analysis_year = st.sidebar.selectbox(
    "Analysis Year",
    [2023, 2024, 2025],
    index=2
)

process_btn = st.sidebar.button("🚀 Process Data")

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<p style='text-align:center;font-size:12px;'>APS Wallet Analytics v2.5</p>",
    unsafe_allow_html=True
)

# =========================
# Session State
# =========================
if "results" not in st.session_state:
    st.session_state.results = None

# =========================
# Process Data
# =========================
if process_btn:
    if onboarding_file and transactions_file:
        with st.spinner("Processing data..."):
            loader = DataLoader()
            analytics = AnalyticsEngine()

            df_onboarding = loader.load_csv(onboarding_file)
            df_transactions = loader.load_csv(transactions_file, nrows=500_000)

            st.session_state.results = analytics.analyze(
                df_onboarding, df_transactions, analysis_year
            )

        st.success("✅ Data processed successfully")
    else:
        st.warning("Please upload both CSV files")

# =========================
# Main Content
# =========================
st.markdown(
    f"""
    <h1 style="text-align:center;">
        APS Wallet Annual Report {analysis_year}
    </h1>
    <p style="text-align:center;color:#6B7280;">
        Last updated: {datetime.now().strftime('%d %B %Y')}
    </p>
    """,
    unsafe_allow_html=True
)

if st.session_state.results:
    r = st.session_state.results

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Active Agents", format_number(r["total_active_agents"]))
    with col2:
        metric_card("Active Tellers", format_number(r["total_active_tellers"]))
    with col3:
        metric_card("Onboarded This Year", format_number(r["onboarded_year"]))
    with col4:
        metric_card("Total Transactions", format_number(r["total_transactions"]))

    st.markdown("### 💰 Transaction Volume")
    st.metric(
        "Total Volume",
        format_currency(r["total_volume"])
    )

    # Service Breakdown
    st.markdown("### 🧾 Service Breakdown")
    fig_service = px.bar(
        r["service_summary"],
        x="Service Name",
        y="sum",
        text="count",
        labels={"sum": "Total Amount"},
        title="Transaction Value by Service"
    )
    st.plotly_chart(fig_service, use_container_width=True)

    # Monthly Trend
    st.markdown("### 📅 Monthly Transaction Trend")
    fig_month = px.line(
        r["monthly_trend"],
        x="month",
        y="Amount",
        markers=True,
        title="Monthly Transaction Volume"
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Downloads
    st.markdown("### 📥 Downloads")

    csv_data = r["service_summary"].to_csv(index=False)
    st.download_button(
        "Download Service Summary (CSV)",
        data=csv_data,
        file_name="service_summary.csv",
        mime="text/csv"
    )

else:
    st.info("Upload data and click **Process Data** to begin.")
