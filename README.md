# Machine Learning Project: Purchase Prediction

## Overview
- Trained 8 ML models for customer purchase prediction
- Implemented SHAP/LIME for model interpretability
- Deployed as Flask REST API

## Files
1. `ml_model_training.py` - Model training and evaluation
2. `model_interpretability.py` - SHAP and LIME analysis
3. `app.py` - Flask API for deployment
4. `model_results/` - Trained models and results
5. `interpretability_results/` - Model explanation visualizations

## Reqirements
  `pip install flask` -Flask==2.3.3
  `pip install pandas as pd` -pandas==2.0.3
  `pip install numpy as np` -numpy==1.24.3
  `pip install scikit-learn` -scikit-learn==1.3.0
  `pip install matplotlib` -matplotlib==3.7.2
  `pip install seaborn` -seaborn==0.12.2
  `pip install shap` -shap==0.42.1
  `pip install lime` -lime==0.2.0.1
  `pip install gunucorn` -gunicorn==20.1.0
  `pip install werkzeug` -Werkzeug==2.3.7
  `pip install joblib` -joblib==1.3.2

## How to Run
1. Train models: `python ml_model_training.py`
2. Interpretability: `python model_interpretability.py`
3. Deploy API: `python app.py`
4. Access at: `http://localhost:5000`

## API Endpoints
- `GET /` - Web interface
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions
