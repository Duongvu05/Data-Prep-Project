"""
Visualization utilities for data storytelling
Contains functions for creating compelling visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# Set style for consistent visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StorytellingVisualizer:
    """Class for creating data storytelling visualizations"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'dark': '#343A40',
            'light': '#F8F9FA'
        }
    
    def compare_data_quality(self, 
                           raw_df: pd.DataFrame, 
                           processed_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization comparing raw vs processed data quality
        
        Args:
            raw_df: Raw dataframe
            processed_df: Processed dataframe
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Comparison: Raw vs Processed Data', fontsize=16, fontweight='bold')
        
        # 1. Missing values comparison
        raw_missing = raw_df.isnull().sum()
        processed_missing = processed_df.isnull().sum()
        
        x_pos = np.arange(len(raw_missing))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, raw_missing.values, width, 
                      label='Raw Data', color=self.colors['danger'], alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, processed_missing.values, width, 
                      label='Processed Data', color=self.colors['success'], alpha=0.7)
        
        axes[0, 0].set_title('Missing Values Comparison')
        axes[0, 0].set_xlabel('Columns')
        axes[0, 0].set_ylabel('Missing Values Count')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(raw_missing.index, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Data shape comparison
        shapes = ['Raw Data', 'Processed Data']
        rows = [raw_df.shape[0], processed_df.shape[0]]
        cols = [raw_df.shape[1], processed_df.shape[1]]
        
        x_pos = np.arange(len(shapes))
        axes[0, 1].bar(x_pos - width/2, rows, width, label='Rows', color=self.colors['primary'], alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, cols, width, label='Columns', color=self.colors['secondary'], alpha=0.7)
        
        axes[0, 1].set_title('Dataset Shape Comparison')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(shapes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Data types distribution
        raw_types = raw_df.dtypes.value_counts()
        processed_types = processed_df.dtypes.value_counts()
        
        axes[1, 0].pie(raw_types.values, labels=raw_types.index, autopct='%1.1f%%', 
                      colors=[self.colors['danger'], self.colors['warning'], self.colors['primary']])
        axes[1, 0].set_title('Raw Data Types Distribution')
        
        axes[1, 1].pie(processed_types.values, labels=processed_types.index, autopct='%1.1f%%',
                      colors=[self.colors['success'], self.colors['accent'], self.colors['secondary']])
        axes[1, 1].set_title('Processed Data Types Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_before_after_distribution(self, 
                                       raw_data: pd.Series, 
                                       processed_data: pd.Series,
                                       title: str = "Distribution Comparison",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create before/after distribution comparison
        
        Args:
            raw_data: Raw data series
            processed_data: Processed data series  
            title: Plot title
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Raw data distribution
        axes[0].hist(raw_data.dropna(), bins=30, alpha=0.7, color=self.colors['danger'], edgecolor='black')
        axes[0].set_title('Raw Data Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics text
        raw_stats = f"Mean: {raw_data.mean():.2f}\nStd: {raw_data.std():.2f}\nMissing: {raw_data.isnull().sum()}"
        axes[0].text(0.02, 0.98, raw_stats, transform=axes[0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Processed data distribution
        axes[1].hist(processed_data.dropna(), bins=30, alpha=0.7, color=self.colors['success'], edgecolor='black')
        axes[1].set_title('Processed Data Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        proc_stats = f"Mean: {processed_data.mean():.2f}\nStd: {processed_data.std():.2f}\nMissing: {processed_data.isnull().sum()}"
        axes[1].text(0.02, 0.98, proc_stats, transform=axes[1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_model_performance_comparison(self, 
                                          results_dict: Dict[str, Dict[str, float]],
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create model performance comparison visualization
        
        Args:
            results_dict: Dictionary with model results {model_name: {metric: value}}
            save_path: Path to save the figure
        """
        # Convert to DataFrame for easier plotting
        df_results = pd.DataFrame(results_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance: Raw vs Processed Data', fontsize=16, fontweight='bold')
        
        metrics = df_results.columns
        colors = [self.colors['danger'], self.colors['success']]
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            if i < 4:  # Only plot first 4 metrics
                bars = axes[row, col].bar(df_results.index, df_results[metric], 
                                        color=colors, alpha=0.7, edgecolor='black')
                
                axes[row, col].set_title(f'{metric.title()} Comparison')
                axes[row, col].set_ylabel(metric.title())
                axes[row, col].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                      f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_comparison_dashboard(self, 
                                              raw_df: pd.DataFrame, 
                                              processed_df: pd.DataFrame,
                                              save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard for data comparison
        
        Args:
            raw_df: Raw dataframe
            processed_df: Processed dataframe
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Values', 'Data Shape', 'Sample Distributions', 'Summary Stats'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # Missing values comparison
        raw_missing = raw_df.isnull().sum()
        processed_missing = processed_df.isnull().sum()
        
        fig.add_trace(
            go.Bar(x=raw_missing.index, y=raw_missing.values, name="Raw Data", 
                  marker_color='red', opacity=0.7),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=processed_missing.index, y=processed_missing.values, name="Processed Data", 
                  marker_color='green', opacity=0.7),
            row=1, col=1
        )
        
        # Data shape comparison
        fig.add_trace(
            go.Bar(x=['Raw Data', 'Processed Data'], 
                  y=[raw_df.shape[0], processed_df.shape[0]], 
                  name="Rows", marker_color='blue', opacity=0.7),
            row=1, col=2
        )
        
        # Sample distribution (first numeric column)
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col_name = numeric_cols[0]
            
            fig.add_trace(
                go.Histogram(x=raw_df[col_name].dropna(), name=f"Raw {col_name}", 
                           opacity=0.7, marker_color='red'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=processed_df[col_name].dropna(), name=f"Processed {col_name}", 
                           opacity=0.7, marker_color='green'),
                row=2, col=1
            )
        
        # Summary statistics table
        summary_data = [
            ['Dataset', 'Rows', 'Columns', 'Missing Values', 'Duplicates'],
            ['Raw', raw_df.shape[0], raw_df.shape[1], raw_df.isnull().sum().sum(), raw_df.duplicated().sum()],
            ['Processed', processed_df.shape[0], processed_df.shape[1], processed_df.isnull().sum().sum(), processed_df.duplicated().sum()]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0], fill_color='lightblue'),
                cells=dict(values=list(zip(*summary_data[1:])), fill_color='lightgray')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Data Preparation Impact Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_storytelling_slide(self, 
                                title: str, 
                                subtitle: str,
                                key_message: str,
                                data_insights: List[str],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a storytelling slide with key insights
        
        Args:
            title: Main title
            subtitle: Subtitle
            key_message: Key message to highlight
            data_insights: List of insights to display
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, title, fontsize=28, fontweight='bold', 
               ha='center', va='center', color=self.colors['dark'])
        
        # Subtitle
        ax.text(0.5, 0.82, subtitle, fontsize=18, 
               ha='center', va='center', color=self.colors['secondary'])
        
        # Key message box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=self.colors['accent'], alpha=0.2)
        ax.text(0.5, 0.65, key_message, fontsize=20, fontweight='bold',
               ha='center', va='center', bbox=bbox_props, color=self.colors['dark'])
        
        # Data insights
        y_start = 0.45
        for i, insight in enumerate(data_insights):
            ax.text(0.1, y_start - i*0.08, f"• {insight}", fontsize=14,
                   ha='left', va='center', color=self.colors['dark'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig