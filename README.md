# AI Fraud Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.0+-green.svg)](https://xgboost.readthedocs.io/en/stable/)

## Overview

This AI-powered fraud detection system utilizes XGBoost machine learning to analyze financial transactions and identify potential fraudulent activities. The application provides a user-friendly web interface built with Streamlit for real-time fraud detection and analysis.

## Features

- Real-time transaction analysis using XGBoost
- High-confidence prediction filtering (85%+ confidence)
- Interactive data visualization
- Batch processing of transaction data
- Exportable prediction results
- Automatic data preprocessing and normalization

## Technical Requirements

### System Requirements

- Python 3.9 or higher
- Modern web browser (Chrome, Firefox, Safari)
- Minimum 8GB RAM recommended

### Software Dependencies

- Streamlit >= 1.0.0
- XGBoost >= 1.0.0
- pandas >= 1.0.0
- scikit-learn >= 0.24.0
- numpy >= 1.19.0

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Fraud-Detection.git
   cd AI-Fraud-Detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Place your pre-trained XGBoost model file (joblib format) in the project directory
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage Guide

1. **Data Upload**
   - Click "Browse files" to select your CSV file containing transaction data
   - Ensure your data format matches the training dataset structure

2. **Prediction Analysis**
   - Click "Predict Fraud" to analyze transactions
   - View high-confidence predictions (85%+ confidence)
   - Monitor potential fraud cases with interactive metrics

3. **Result Export**
   - Download prediction results as CSV
   - Export includes transaction details and fraud probabilities

## Input Data Format

The application expects a CSV file containing the following features:
- Step (time step)
- Amount (transaction amount)
- Old balance origin
- New balance origin
- Old balance destination
- New balance destination
- Log amount (log-transformed transaction amount)

## Output Format

The application generates:
- High-confidence predictions (50%+ confidence)
- Fraud probability scores
- Prediction labels (0: Not Fraud, 1: Fraud)
- Exportable results in CSV format

## Performance Metrics

The model is optimized for:
- High precision in fraud detection
- Low false positive rate
- Fast prediction times
- Scalable batch processing

## Security Considerations

- All data processing is performed locally
- No data is stored or transmitted without explicit permission
- Secure model loading and execution

## Contributers

Meray basanty     2205060@anu.edu.eg
Mohamed Ghonem    2205050@anu.edu.eg
Malak hashim      2205090@anu.edu.eg

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the project maintainer.
