# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a blood glucose level prediction project for diabetes management using time series data and deep learning. The project implements a hybrid neural network combining LSTM layers for time series processing with cross-attention mechanisms to predict blood glucose levels 15, 30, 45, and 60 minutes into the future.

## Common Development Commands

### Data Preprocessing
```bash
# Convert Excel files to CSV
python pre-process/trans_xlsx2csv.py

# Extract and process medication agents
python pre-process/agents_process.py

# Generate attributes and process all raw data
python pre-process/generate_attribute.py
python pre-process/pre_process.py

# Split processed datasets
python pre-process/split_datasets.py
```

### Model Training and Evaluation
```bash
# Train the blood glucose prediction model
python model.py

# Validate the trained model
python model_vaild.py

# Test model on full dataset
python model_test.py
```

### Model Performance Analysis
```bash
# Generate MAE curves from training logs
python validation/MAECurve/MAE_curve.py

# Calculate MSE and MAE on predictions
python validation/MSE-MAE/cal_MSE_MAE.py

# Compare predictions vs actual values
python validation/CrossValidation/pred_real_compare.py
```

## Architecture Overview

### Data Pipeline
1. **Raw Data**: Shanghai_T1DM and Shanghai_T2DM datasets (Excel format)
2. **Preprocessing**: Complex medical data processing including:
   - Medication agent extraction and categorization
   - Insulin dose processing (subcutaneous, intravenous, CSII)
   - Dietary intake binary encoding
   - Time series feature engineering
3. **Processed Data**: Cleaned CSV files in `pre-process/processed-data/`
4. **Model Input**: Time series features + static patient features

### Model Architecture
- **Time Series Encoder**: Multi-layer LSTM (64→56→48→40→36→32 units)
- **Static Features Encoder**: Dense layers (64→56→48→40→36→32)
- **Cross-Attention**: Bidirectional attention between time series and static embeddings
- **Decoder**: Dense layers gradually reducing to 4 outputs (15/30/45/60 min predictions)

### Key Files Structure
- `pre-process/`: Data preprocessing pipeline and utilities
  - `raw-data/`: Original Shanghai T1DM/T2DM Excel files
  - `processed-data/`: Cleaned CSV files after preprocessing
  - `*.json`: Configuration files for feature attributes
- `model.py`: Main model training and architecture
- `model_test.py`: Full dataset prediction and evaluation
- `validation/`: Performance analysis and visualization tools
- `GCM_model.h5`: Trained model weights (generated after training)

## Data Processing Notes

### Medical Data Complexity
The project handles complex medical data including:
- Multiple insulin types and administration methods
- CSII (continuous subcutaneous insulin infusion) data
- Non-insulin hypoglycemic agents
- Dietary intake records
- Blood glucose measurements (CGM and CBG)

### Feature Engineering
- Time series attributes include CGM readings, medication doses, dietary intake
- Static attributes include patient demographics and clinical characteristics
- Target variables: blood glucose levels at 15, 30, 45, 60 minutes

### Data Quality
Some files are excluded due to format issues:
- `2045_0_20201216.csv`, `2095_0_20201116.csv` (missing insulin units)
- `2013_0_20220123.csv` (no dietary data, only quantities)
- `2027_0_20210521.csv` (Chinese insulin dose format)

## Model Training Environment

The model was trained on AutoDL cloud platform with:
- GPU: RTX 4090D (24GB)
- CPU: 15 vCPU Intel Xeon Platinum 8474C
- Environment: TensorFlow 2.9.0, Python 3.8 (Ubuntu 20.04), CUDA 11.2

## Important Processing Details

### Time Series Sequences
Model uses 10-timestep windows for sequence prediction, creating overlapping sequences from the continuous glucose monitoring data.

### Cross-Attention Mechanism
Implements bidirectional attention:
1. Static features query time series features
2. Time series features query static features
Results are concatenated and processed through decoder layers.

### Data Standardization
All features and targets are standardized using StandardScaler:
- Separate scalers for time series features, static features, and targets
- Scalers are saved with the model for consistent inference

## Performance Evaluation
The model achieves approximately MAE = 0.25 (normalized) on test data, with comprehensive evaluation including:
- Training progress monitoring
- Cross-validation on full dataset
- Residual analysis and percentage error calculations
- MAE/MSE metrics for each time horizon (15/30/45/60 minutes)