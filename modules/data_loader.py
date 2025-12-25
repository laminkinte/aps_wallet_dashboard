"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self):
        self.processed_data = {}
    
    def load_onboarding_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess onboarding data"""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Required columns
            required_cols = ['Account ID', 'Entity', 'Status', 'Registration Date']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Clean data
            df['Entity'] = df['Entity'].astype(str).str.upper().str.strip()
            df['Status'] = df['Status'].astype(str).str.upper().str.strip()
            df['Account ID'] = df['Account ID'].astype(str).str.strip()
            
            # Parse dates
            df['Registration Date'] = pd.to_datetime(
                df['Registration Date'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            )
            
            logger.info(f"Loaded {len(df)} onboarding records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading onboarding data: {e}")
            raise
    
    def load_transaction_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and preprocess transaction data"""
        try:
            if sample_size:
                # Read in chunks for large files
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=100000):
                    chunks.append(chunk)
                    if sum(len(c) for c in chunks) >= sample_size:
                        break
                df = pd.concat(chunks, ignore_index=True).head(sample_size)
            else:
                df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Clean data
            text_columns = ['Entity Name', 'Service Name', 'Transaction Type', 'Product Name']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.upper().str.strip()
            
            # Parse dates
            if 'Created At' in df.columns:
                df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
            
            # Clean identifiers
            if 'User Identifier' in df.columns:
                df['User Identifier'] = pd.to_numeric(
                    df['User Identifier'].astype(str).str.replace(r'\D', '', regex=True),
                    errors='coerce'
                ).astype('Int64')
            
            if 'Parent User Identifier' in df.columns:
                df['Parent User Identifier'] = pd.to_numeric(
                    df['Parent User Identifier'].astype(str).str.replace(r'\D', '', regex=True),
                    errors='coerce'
                ).astype('Int64')
            
            logger.info(f"Loaded {len(df)} transaction records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading transaction data: {e}")
            raise
    
    def validate_data(self, onboarding_df: pd.DataFrame, transaction_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the loaded data"""
        try:
            # Check for required columns
            onboarding_required = ['Account ID', 'Entity', 'Status', 'Registration Date']
            transaction_required = ['User Identifier', 'Created At']
            
            for col in onboarding_required:
                if col not in onboarding_df.columns:
                    return False, f"Missing column in onboarding data: {col}"
            
            for col in transaction_required:
                if col not in transaction_df.columns:
                    return False, f"Missing column in transaction data: {col}"
            
            # Check data quality
            if onboarding_df['Account ID'].isna().any():
                return False, "Missing Account IDs in onboarding data"
            
            if transaction_df['Created At'].isna().all():
                return False, "No valid dates in transaction data"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"