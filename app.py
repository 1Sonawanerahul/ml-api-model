"""
FLASK API FOR MODEL DEPLOYMENT
Simple REST API for model predictions
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_model():
    """Load the trained model and scaler"""
    global model, scaler, feature_names
    
    try:
        # Load model
        with open('model_results/random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('model_results/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open('model_results/features.json', 'r') as f:
            feature_names = json.load(f)
        
        print("✓ Model loaded successfully")
        print(f"✓ Features: {len(feature_names)}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame with correct feature names
        df = pd.DataFrame([data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[feature_names]
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        return scaled_data, None
        
    except Exception as e:
        return None, str(e)

def create_template():
    """Create HTML template for home page"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Deployment API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); width: 100%; max-width: 1200px; overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
        @media (max-width: 768px) { .content { grid-template-columns: 1fr; } }
        .card { background: #f8f9fa; border-radius: 15px; padding: 30px; border-left: 5px solid #667eea; transition: transform 0.3s ease; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .card h2 { color: #333; margin-bottom: 20px; font-size: 1.5em; }
        .endpoint { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #e0e0e0; }
        .method { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9em; margin-right: 10px; }
        .get { background: #61affe; color: white; }
        .post { background: #49cc90; color: white; }
        .url { font-family: monospace; background: #f1f1f1; padding: 8px 15px; border-radius: 5px; margin: 10px 0; display: block; word-break: break-all; }
        .example { background: #1a1a1a; color: #00ff00; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em; overflow-x: auto; margin-top: 10px; }
        .try-section { background: white; border-radius: 15px; padding: 30px; margin-top: 20px; }
        .try-section h3 { margin-bottom: 20px; color: #333; }
        .input-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        input, textarea { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 1em; transition: border-color 0.3s ease; }
        input:focus, textarea:focus { outline: none; border-color: #667eea; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: transform 0.3s ease; width: 100%; }
        button:hover { transform: translateY(-2px); }
        .response { background: #f1f1f1; border-radius: 8px; padding: 20px; margin-top: 20px; display: none; }
        .status { padding: 5px 15px; border-radius: 20px; font-weight: bold; margin-bottom: 10px; display: inline-block; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .footer { text-align: center; padding: 20px; background: #f8f9fa; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ML Model Deployment API</h1>
            <p>Production-ready machine learning model serving</p>
        </div>
        
        <div class="content">
            <div>
                <div class="card">
                    <h2>📊 API Endpoints</h2>
                    <div class="endpoint"><span class="method get">GET</span><strong>Health Check</strong><span class="url">/health</span><p>Check if API is running and model is loaded.</p></div>
                    <div class="endpoint"><span class="method post">POST</span><strong>Single Prediction</strong><span class="url">/predict</span><p>Make prediction for single sample.</p>
                        <div class="example">{ "Age": 30.5, "Salary": 50000, "Gender_Male": 1, "Spending_Power_Index": 1.2 }</div>
                    </div>
                    <div class="endpoint"><span class="method post">POST</span><strong>Batch Predictions</strong><span class="url">/batch_predict</span><p>Make predictions for multiple samples.</p></div>
                    <div class="endpoint"><span class="method get">GET</span><strong>Model Info</strong><span class="url">/model_info</span><p>Get model details and feature information.</p></div>
                </div>
                
                <div class="card">
                    <h2>⚙️ Model Information</h2>
                    <p><strong>Model Type:</strong> Random Forest Classifier</p>
                    <p><strong>Purpose:</strong> Purchase Prediction</p>
                    <p><strong>Features:</strong> <span id="featuresCount">Loading...</span></p>
                    <p><strong>Status:</strong> <span id="modelStatus">Loading...</span></p>
                    <button onclick="checkHealth()" style="margin-top: 20px;">Check Health</button>
                    <div id="healthResponse" class="response"></div>
                </div>
            </div>
            
            <div>
                <div class="try-section">
                    <h3>🔮 Try Prediction API</h3>
                    <div class="input-group"><label for="feature1">Age:</label><input type="number" id="feature1" value="30.5" step="0.1"></div>
                    <div class="input-group"><label for="feature2">Salary:</label><input type="number" id="feature2" value="50000"></div>
                    <div class="input-group"><label for="feature3">Gender_Male (0/1):</label><input type="number" id="feature3" value="1" min="0" max="1"></div>
                    <div class="input-group"><label for="feature4">Spending_Power_Index:</label><input type="number" id="feature4" value="1.2" step="0.1"></div>
                    <button onclick="makePrediction()">Make Prediction</button>
                    <div id="predictionResponse" class="response"><div class="status success">Success</div><pre id="responseContent"></pre></div>
                </div>
                
                <div class="try-section">
                    <h3>📝 Test with Raw JSON</h3>
                    <textarea id="jsonInput" rows="8" placeholder='Enter JSON data...'>{"Age": 30.5, "Salary": 50000, "Gender_Male": 1, "Spending_Power_Index": 1.2}</textarea>
                    <button onclick="testWithJson()" style="margin-top: 15px;">Test API</button>
                    <div id="jsonResponse" class="response"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Machine Learning Model Deployment API | Built with Flask</p>
            <p>For documentation and examples, visit the API endpoints above.</p>
        </div>
    </div>
    
    <script>
        fetch('/model_info').then(r => r.json()).then(data => {
            if(data.feature_count) {
                document.getElementById('featuresCount').textContent = data.feature_count;
                document.getElementById('modelStatus').innerHTML = '✅ Loaded';
                document.getElementById('modelStatus').style.color = 'green';
            }
        }).catch(e => {
            document.getElementById('modelStatus').innerHTML = '❌ Error';
            document.getElementById('modelStatus').style.color = 'red';
        });
        
        function checkHealth() {
            fetch('/health').then(r => r.json()).then(data => {
                const div = document.getElementById('healthResponse');
                div.style.display = 'block';
                div.innerHTML = `<div class="status success">Healthy ✅</div><pre>${JSON.stringify(data, null, 2)}</pre>`;
            }).catch(e => {
                const div = document.getElementById('healthResponse');
                div.style.display = 'block';
                div.innerHTML = `<div class="status error">Error ❌</div><pre>${e.toString()}</pre>`;
            });
        }
        
        function makePrediction() {
            const data = {
                Age: parseFloat(document.getElementById('feature1').value),
                Salary: parseFloat(document.getElementById('feature2').value),
                Gender_Male: parseInt(document.getElementById('feature3').value),
                Spending_Power_Index: parseFloat(document.getElementById('feature4').value)
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(r => r.json()).then(data => {
                const div = document.getElementById('predictionResponse');
                const content = document.getElementById('responseContent');
                div.style.display = 'block';
                content.textContent = JSON.stringify(data, null, 2);
                const status = div.querySelector('.status');
                if(data.error) {
                    status.className = 'status error';
                    status.textContent = 'Error ❌';
                } else {
                    status.className = 'status success';
                    status.textContent = `Prediction: ${data.prediction_label} (${(data.confidence*100).toFixed(1)}% confidence) ✅`;
                }
            }).catch(e => {
                const div = document.getElementById('predictionResponse');
                const content = document.getElementById('responseContent');
                div.style.display = 'block';
                content.textContent = e.toString();
                div.querySelector('.status').className = 'status error';
                div.querySelector('.status').textContent = 'Error ❌';
            });
        }
        
        function testWithJson() {
            try {
                const jsonData = JSON.parse(document.getElementById('jsonInput').value);
                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(jsonData)
                }).then(r => r.json()).then(data => {
                    const div = document.getElementById('jsonResponse');
                    div.style.display = 'block';
                    div.innerHTML = `<div class="status ${data.error?'error':'success'}">${data.error?'Error ❌':'Success ✅'}</div><pre>${JSON.stringify(data,null,2)}</pre>`;
                }).catch(e => {
                    const div = document.getElementById('jsonResponse');
                    div.style.display = 'block';
                    div.innerHTML = `<div class="status error">Error ❌</div><pre>${e.toString()}</pre>`;
                });
            } catch(e) {
                const div = document.getElementById('jsonResponse');
                div.style.display = 'block';
                div.innerHTML = `<div class="status error">JSON Parse Error ❌</div><pre>${e.toString()}</pre>`;
            }
        }
    </script>
</body>
</html>"""
    
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        processed_data, error = preprocess_input(data)
        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 400
        
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Purchased' if prediction == 1 else 'Not Purchased',
            'confidence': float(max(prediction_proba)),
            'probabilities': {
                'not_purchased': float(prediction_proba[0]),
                'purchased': float(prediction_proba[1])
            },
            'timestamp': datetime.now().isoformat(),
            'model': 'Random Forest Classifier'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch predictions"""
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        results = []
        
        for sample in samples:
            processed_data, error = preprocess_input(sample)
            if error:
                results.append({'error': error, 'sample': sample})
                continue
            
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            results.append({
                'prediction': int(prediction),
                'prediction_label': 'Purchased' if prediction == 1 else 'Not Purchased',
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    'not_purchased': float(prediction_proba[0]),
                    'purchased': float(prediction_proba[1])
                }
            })
        
        response = {
            'results': results,
            'total_samples': len(samples),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': type(model).__name__,
            'features': feature_names,
            'feature_count': len(feature_names),
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'loaded_at': datetime.now().isoformat()
        }
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]
            info['top_features'] = [{'feature': f, 'importance': float(i)} for f, i in top_features]
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    if load_model():
        print("="*60)
        print("ML MODEL DEPLOYMENT API")
        print("="*60)
        print("Model loaded successfully!")
        print(f"Features: {len(feature_names)}")
        print("\nAvailable endpoints:")
        print("  GET  /              - Home page")
        print("  GET  /health        - Health check")
        print("  POST /predict       - Single prediction")
        print("  POST /batch_predict - Batch predictions")
        print("  GET  /model_info    - Model information")
        print("\nStarting server on http://localhost:5000")
        print("="*60)
        
        # Create template
        create_template()
        
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Exiting...")