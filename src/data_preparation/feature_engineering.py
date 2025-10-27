"""
Feature Engineering Module
Contains functions for creating and transforming features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Tuple, Optional, Union


class FeatureEngineer:
    """Class for feature engineering operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}
        self.feature_log = []
    
    def create_datetime_features(self, date_columns: List[str]) -> 'FeatureEngineer':
        """
        Create datetime-based features
        
        Args:
            date_columns: List of datetime column names
        """
        for col in date_columns:
            if col in self.df.columns:
                # Ensure column is datetime
                self.df[col] = pd.to_datetime(self.df[col])
                
                # Extract features
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_day'] = self.df[col].dt.day
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                self.df[f'{col}_quarter'] = self.df[col].dt.quarter
                self.df[f'{col}_is_weekend'] = (self.df[col].dt.dayofweek >= 5).astype(int)
                
                self.feature_log.append(f"Created datetime features from {col}")
        
        return self
    
    def create_interaction_features(self, 
                                  feature_pairs: List[Tuple[str, str]],
                                  operations: List[str] = ['multiply']) -> 'FeatureEngineer':
        """
        Create interaction features between pairs of numeric columns
        
        Args:
            feature_pairs: List of tuples containing column pairs
            operations: List of operations ('multiply', 'add', 'subtract', 'divide')
        """
        for col1, col2 in feature_pairs:
            if col1 in self.df.columns and col2 in self.df.columns:
                if self.df[col1].dtype in ['int64', 'float64'] and self.df[col2].dtype in ['int64', 'float64']:
                    
                    if 'multiply' in operations:
                        self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
                    
                    if 'add' in operations:
                        self.df[f'{col1}_plus_{col2}'] = self.df[col1] + self.df[col2]
                    
                    if 'subtract' in operations:
                        self.df[f'{col1}_minus_{col2}'] = self.df[col1] - self.df[col2]
                    
                    if 'divide' in operations:
                        # Avoid division by zero
                        self.df[f'{col1}_div_{col2}'] = self.df[col1] / (self.df[col2] + 1e-8)
                    
                    self.feature_log.append(f"Created interaction features between {col1} and {col2}")
        
        return self
    
    def create_binning_features(self, 
                               numeric_columns: List[str],
                               n_bins: int = 5,
                               strategy: str = 'quantile') -> 'FeatureEngineer':
        """
        Create binned categorical features from numeric columns
        
        Args:
            numeric_columns: List of numeric column names
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile')
        """
        for col in numeric_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                if strategy == 'quantile':
                    self.df[f'{col}_binned'] = pd.qcut(self.df[col], q=n_bins, labels=False, duplicates='drop')
                else:  # uniform
                    self.df[f'{col}_binned'] = pd.cut(self.df[col], bins=n_bins, labels=False)
                
                self.feature_log.append(f"Created binned feature from {col}")
        
        return self
    
    def encode_categorical_features(self, 
                                  categorical_columns: List[str],
                                  encoding_type: str = 'onehot',
                                  drop_first: bool = True) -> 'FeatureEngineer':
        """
        Encode categorical features
        
        Args:
            categorical_columns: List of categorical column names
            encoding_type: Type of encoding ('onehot', 'label')
            drop_first: Whether to drop first category in one-hot encoding
        """
        for col in categorical_columns:
            if col in self.df.columns:
                if encoding_type == 'onehot':
                    # One-hot encoding
                    encoded = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first)
                    self.df = pd.concat([self.df.drop(col, axis=1), encoded], axis=1)
                    self.feature_log.append(f"One-hot encoded {col}")
                
                elif encoding_type == 'label':
                    # Label encoding
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.encoders[col] = le
                    self.feature_log.append(f"Label encoded {col}")
        
        return self
    
    def scale_features(self, 
                      numeric_columns: List[str],
                      scaling_type: str = 'standard') -> 'FeatureEngineer':
        """
        Scale numeric features
        
        Args:
            numeric_columns: List of numeric column names
            scaling_type: Type of scaling ('standard', 'minmax')
        """
        for col in numeric_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                if scaling_type == 'standard':
                    scaler = StandardScaler()
                elif scaling_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    continue
                
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
                self.feature_log.append(f"Scaled {col} using {scaling_type} scaling")
        
        return self
    
    def select_features(self, 
                       target_column: str,
                       k: int = 10,
                       score_func = f_classif) -> 'FeatureEngineer':
        """
        Select top k features based on statistical tests
        
        Args:
            target_column: Name of target column
            k: Number of features to select
            score_func: Statistical test function
        """
        if target_column in self.df.columns:
            # Separate features and target
            X = self.df.drop(target_column, axis=1).select_dtypes(include=[np.number])
            y = self.df[target_column]
            
            # Select features
            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Keep only selected features plus target
            self.df = self.df[selected_features + [target_column]]
            self.feature_log.append(f"Selected top {len(selected_features)} features")
        
        return self
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Return the dataframe with engineered features"""
        return self.df
    
    def get_feature_summary(self) -> Dict:
        """Return summary of feature engineering operations"""
        return {
            'final_shape': self.df.shape,
            'feature_engineering_steps': self.feature_log,
            'encoders_used': list(self.encoders.keys()),
            'scalers_used': list(self.scalers.keys())
        }