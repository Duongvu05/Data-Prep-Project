"""
Data Preparation Utilities Module
Provides common functions for data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats
    
    Args:
        file_path (str): Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path, **kwargs)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path, **kwargs)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data quality report
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict: Data quality metrics
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    return report


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range method
    
    Args:
        series (pd.Series): Input series
        factor (float): IQR multiplication factor
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save dataframe to various file formats
    
    Args:
        df (pd.DataFrame): Dataframe to save
        file_path (str): Output file path
        **kwargs: Additional arguments for pandas save functions
    """
    try:
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False, **kwargs)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False, **kwargs)
        elif file_path.endswith('.json'):
            df.to_json(file_path, **kwargs)
        elif file_path.endswith('.parquet'):
            df.to_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise