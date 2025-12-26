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
