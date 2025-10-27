"""
Machine Learning Models for comparison between raw and processed data
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class ModelComparator:
    """Class for comparing model performance on raw vs processed data"""
    
    def __init__(self):
        self.models = {
            'classification': {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
            },
            'regression': {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression()
            }
        }
        self.results = {}
    
    def prepare_data_for_modeling(self, 
                                df: pd.DataFrame, 
                                target_column: str,
                                test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle missing values (simple imputation)
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y if self._is_classification(y) else None
        )
        
        return X_train, X_test, y_train, y_test
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if task is classification or regression"""
        return y.dtype == 'object' or y.nunique() <= 10
    
    def evaluate_classification_model(self, 
                                    model, 
                                    X_train: np.ndarray, 
                                    X_test: np.ndarray,
                                    y_train: np.ndarray, 
                                    y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            model: Trained model
            X_train, X_test, y_train, y_test: Train/test splits
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_accuracy'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def evaluate_regression_model(self, 
                                model, 
                                X_train: np.ndarray, 
                                X_test: np.ndarray,
                                y_train: np.ndarray, 
                                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            model: Trained model
            X_train, X_test, y_train, y_test: Train/test splits
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def compare_datasets(self, 
                        raw_df: pd.DataFrame, 
                        processed_df: pd.DataFrame,
                        target_column: str,
                        model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare model performance between raw and processed datasets
        
        Args:
            raw_df: Raw dataset
            processed_df: Processed dataset
            target_column: Name of target column
            model_names: List of model names to use
            
        Returns:
            Nested dictionary with results
        """
        results = {
            'raw_data': {},
            'processed_data': {}
        }
        
        # Determine task type
        is_classification = self._is_classification(raw_df[target_column])
        task_type = 'classification' if is_classification else 'regression'
        
        # Select models
        models_to_use = self.models[task_type]
        if model_names:
            models_to_use = {k: v for k, v in models_to_use.items() if k in model_names}
        
        # Evaluate on raw data
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = self.prepare_data_for_modeling(
                raw_df, target_column
            )
            
            for model_name, model in models_to_use.items():
                if is_classification:
                    metrics = self.evaluate_classification_model(
                        model, X_train_raw, X_test_raw, y_train_raw, y_test_raw
                    )
                else:
                    metrics = self.evaluate_regression_model(
                        model, X_train_raw, X_test_raw, y_train_raw, y_test_raw
                    )
                
                results['raw_data'][model_name] = metrics
        
        except Exception as e:
            print(f"Error evaluating raw data: {str(e)}")
            results['raw_data'] = {}
        
        # Evaluate on processed data
        try:
            X_train_proc, X_test_proc, y_train_proc, y_test_proc = self.prepare_data_for_modeling(
                processed_df, target_column
            )
            
            for model_name, model in models_to_use.items():
                if is_classification:
                    metrics = self.evaluate_classification_model(
                        model, X_train_proc, X_test_proc, y_train_proc, y_test_proc
                    )
                else:
                    metrics = self.evaluate_regression_model(
                        model, X_train_proc, X_test_proc, y_train_proc, y_test_proc
                    )
                
                results['processed_data'][model_name] = metrics
        
        except Exception as e:
            print(f"Error evaluating processed data: {str(e)}")
            results['processed_data'] = {}
        
        self.results = results
        return results
    
    def get_improvement_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate improvement percentages from raw to processed data
        
        Returns:
            Dictionary with improvement percentages
        """
        if not self.results:
            return {}
        
        improvements = {}
        
        for model_name in self.results['raw_data']:
            if model_name in self.results['processed_data']:
                improvements[model_name] = {}
                
                raw_metrics = self.results['raw_data'][model_name]
                proc_metrics = self.results['processed_data'][model_name]
                
                for metric in raw_metrics:
                    if metric in proc_metrics and raw_metrics[metric] != 0:
                        improvement = ((proc_metrics[metric] - raw_metrics[metric]) / 
                                     raw_metrics[metric]) * 100
                        improvements[model_name][metric] = improvement
        
        return improvements
    
    def get_best_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Get best performing model for each dataset
        
        Returns:
            Dictionary with best model results
        """
        if not self.results:
            return {}
        
        best_comparison = {}
        
        for dataset in ['raw_data', 'processed_data']:
            if self.results[dataset]:
                # Find best model based on primary metric
                best_model = None
                best_score = float('-inf')
                
                for model_name, metrics in self.results[dataset].items():
                    # Use accuracy for classification, r2_score for regression
                    primary_metric = 'accuracy' if 'accuracy' in metrics else 'r2_score'
                    
                    if primary_metric in metrics and metrics[primary_metric] > best_score:
                        best_score = metrics[primary_metric]
                        best_model = model_name
                
                if best_model:
                    best_comparison[dataset] = {
                        'model': best_model,
                        **self.results[dataset][best_model]
                    }
        
        return best_comparison