"""
Data Cleaning and Preprocessing Module
Contains functions for cleaning and preparing raw data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Class for data cleaning operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
    
    def handle_missing_values(self, 
                            strategy: Dict[str, str] = None,
                            default_strategy: str = 'drop') -> 'DataCleaner':
        """
        Handle missing values based on specified strategies
        
        Args:
            strategy: Dict mapping column names to strategies ('drop', 'mean', 'median', 'mode', 'forward_fill')
            default_strategy: Default strategy for columns not specified
        """
        if strategy is None:
            strategy = {}
        
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                col_strategy = strategy.get(column, default_strategy)
                
                if col_strategy == 'drop':
                    before_count = len(self.df)
                    self.df = self.df.dropna(subset=[column])
                    after_count = len(self.df)
                    self.cleaning_log.append(f"Dropped {before_count - after_count} rows due to missing {column}")
                
                elif col_strategy == 'mean' and self.df[column].dtype in ['int64', 'float64']:
                    fill_value = self.df[column].mean()
                    self.df[column].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled missing {column} with mean: {fill_value:.2f}")
                
                elif col_strategy == 'median' and self.df[column].dtype in ['int64', 'float64']:
                    fill_value = self.df[column].median()
                    self.df[column].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled missing {column} with median: {fill_value:.2f}")
                
                elif col_strategy == 'mode':
                    fill_value = self.df[column].mode()[0] if not self.df[column].mode().empty else 'Unknown'
                    self.df[column].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled missing {column} with mode: {fill_value}")
                
                elif col_strategy == 'forward_fill':
                    self.df[column].fillna(method='ffill', inplace=True)
                    self.cleaning_log.append(f"Forward filled missing values in {column}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        """Remove duplicate rows"""
        before_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        after_count = len(self.df)
        removed = before_count - after_count
        
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} duplicate rows")
        
        return self
    
    def handle_outliers(self, 
                       columns: List[str], 
                       method: str = 'iqr',
                       action: str = 'remove') -> 'DataCleaner':
        """
        Handle outliers in specified columns
        
        Args:
            columns: List of column names
            method: Method to detect outliers ('iqr', 'zscore')
            action: Action to take ('remove', 'cap')
        """
        for column in columns:
            if column not in self.df.columns or self.df[column].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                outliers_mask = z_scores > 3
            
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                if action == 'remove':
                    self.df = self.df[~outliers_mask]
                    self.cleaning_log.append(f"Removed {outliers_count} outliers from {column}")
                
                elif action == 'cap':
                    if method == 'iqr':
                        self.df.loc[self.df[column] < lower_bound, column] = lower_bound
                        self.df.loc[self.df[column] > upper_bound, column] = upper_bound
                        self.cleaning_log.append(f"Capped {outliers_count} outliers in {column}")
        
        return self
    
    def standardize_column_names(self) -> 'DataCleaner':
        """Standardize column names (lowercase, replace spaces with underscores)"""
        old_columns = self.df.columns.tolist()
        new_columns = [col.lower().replace(' ', '_').replace('-', '_') for col in old_columns]
        self.df.columns = new_columns
        
        if old_columns != new_columns:
            self.cleaning_log.append("Standardized column names")
        
        return self
    
    def convert_data_types(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        """Convert data types of specified columns"""
        for column, new_type in type_mapping.items():
            if column in self.df.columns:
                try:
                    if new_type == 'datetime':
                        self.df[column] = pd.to_datetime(self.df[column])
                    else:
                        self.df[column] = self.df[column].astype(new_type)
                    
                    self.cleaning_log.append(f"Converted {column} to {new_type}")
                
                except Exception as e:
                    logger.warning(f"Could not convert {column} to {new_type}: {str(e)}")
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe"""
        return self.df
    
    def get_cleaning_summary(self) -> Dict:
        """Return summary of cleaning operations"""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'cleaning_steps': self.cleaning_log
        }