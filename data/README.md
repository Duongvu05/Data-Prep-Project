# Data folder README

## Data Directory Structure

This directory contains the datasets used in the Data Preparation project.

### Subdirectories

#### `raw/`
- Contains original, unprocessed datasets
- These files should not be modified
- Include data dictionaries and source information

#### `processed/`
- Contains cleaned, preprocessed datasets
- Results of data preparation pipeline
- Ready for analysis and modeling

### Data Guidelines

1. **Raw Data**
   - Keep original files unchanged
   - Document data sources and collection methods
   - Include any available metadata or data dictionaries

2. **Processed Data**
   - Save intermediate and final processed versions
   - Use clear, descriptive filenames
   - Include version numbers if multiple iterations exist

3. **File Formats**
   - Prefer CSV for tabular data
   - Use Parquet for large datasets
   - JSON for nested/structured data
   - Include README files describing each dataset

### Example Structure
```
data/
├── raw/
│   ├── dataset_name_raw.csv
│   ├── data_dictionary.txt
│   └── source_info.md
└── processed/
    ├── dataset_name_cleaned.csv
    ├── dataset_name_features.csv
    └── processing_log.txt
```

### Data Privacy and Security
- Ensure sensitive data is properly anonymized
- Follow institutional data handling policies
- Do not commit large data files to version control
- Use .gitignore to exclude data files if necessary