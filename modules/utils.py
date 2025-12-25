"""
Utility functions for the dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json

def format_number(value: Any) -> str:
    """Format number with commas"""
    if isinstance(value, (int, float)):
        return f"{value:,.0f}"
    return str(value)

def format_currency(value: float) -> str:
    """Format currency value"""
    return f"GMD {value:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:.1f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Filter DataFrame based on criteria"""
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns:
            if isinstance(value, (list, tuple)):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def calculate_date_range(period: str) -> tuple:
    """Calculate date range based on period"""
    end_date = datetime.now()
    
    if period == 'last_7_days':
        start_date = end_date - timedelta(days=7)
    elif period == 'last_30_days':
        start_date = end_date - timedelta(days=30)
    elif period == 'last_90_days':
        start_date = end_date - timedelta(days=90)
    elif period == 'last_year':
        start_date = end_date - timedelta(days=365)
    else:
        start_date = datetime(end_date.year, 1, 1)  # Year to date
    
    return start_date, end_date

def generate_sample_data(size: int = 100) -> pd.DataFrame:
    """Generate sample data for testing"""
    data = {
        'date': pd.date_range('2025-01-01', periods=size, freq='D'),
        'value': np.random.normal(100, 20, size),
        'category': np.random.choice(['A', 'B', 'C'], size)
    }
    
    return pd.DataFrame(data)

def validate_email(email: str) -> bool:
    """Validate email address"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def log_message(message: str, level: str = 'INFO'):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")