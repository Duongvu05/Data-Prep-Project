# Sample configuration file for the project

# Data paths
DATA_PATHS = {
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed/',
    'outputs': 'outputs/'
}

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

# Data preparation settings
DATA_PREP_CONFIG = {
    'missing_value_threshold': 0.5,  # Drop columns with >50% missing values
    'outlier_method': 'iqr',
    'outlier_factor': 1.5,
    'feature_selection_k': 10
}

# Storytelling elements
STORYTELLING_CONFIG = {
    'primary_color': '#2E86AB',
    'secondary_color': '#A23B72',
    'accent_color': '#F18F01',
    'success_color': '#28A745',
    'warning_color': '#FFC107',
    'danger_color': '#DC3545'
}