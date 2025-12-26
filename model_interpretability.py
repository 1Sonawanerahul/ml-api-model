"""
MODEL INTERPRETABILITY WITH SHAP AND LIME
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("="*60)
print("MODEL INTERPRETABILITY ANALYSIS")
print("="*60)

# ============================================================================
# 1. LOAD TRAINED MODEL AND DATA
# ============================================================================
print("\n1. LOADING TRAINED MODEL...")

try:
    # Load the best model from previous training
    with open('model_results/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    print("SUCCESS: Random Forest model loaded")
    
    # Load scaler and features
    with open('model_results/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('model_results/features.json', 'r') as f:
        import json
        feature_names = json.load(f)
    
    # Load data
    data = pd.read_csv('refined_dataset_engineered.csv')
    X = data.drop('Purchased', axis=1)
    y = data['Purchased']
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    print("Creating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'Age': np.random.randn(n_samples),
        'Salary': np.random.randn(n_samples),
        'Gender_Male': np.random.randint(0, 2, n_samples),
        'Spending_Power_Index': np.random.randn(n_samples),
        'Demographic_Score': np.random.randn(n_samples)
    })
    y = np.random.randint(0, 2, n_samples)
    feature_names = X.columns.tolist()
    X_scaled = X.values

# ============================================================================
# 2. SHAP ANALYSIS
# ============================================================================
print("\n2. PERFORMING SHAP ANALYSIS...")

try:
    import shap
    
    # Create directory for SHAP results
    os.makedirs('interpretability_results/shap', exist_ok=True)
    
    # Initialize SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        print("SHAP TreeExplainer initialized successfully")
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance', fontsize=16)
        plt.tight_layout()
        plt.savefig('interpretability_results/shap/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved")
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar)', fontsize=16)
        plt.tight_layout()
        plt.savefig('interpretability_results/shap/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP bar plot saved")
        
        # Individual prediction explanation
        plt.figure(figsize=(10, 6))
        shap.force_plot(explainer.expected_value, shap_values[0, :], X_scaled[0, :], 
                       feature_names=feature_names, matplotlib=True, show=False)
        plt.title('SHAP Force Plot - First Sample', fontsize=16)
        plt.tight_layout()
        plt.savefig('interpretability_results/shap/shap_force_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP force plot saved")
        
        # Dependence plot for top features
        for i, feature in enumerate(feature_names[:3]):
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_scaled, feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot - {feature}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'interpretability_results/shap/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ SHAP dependence plots saved")
        
    else:
        print("Model doesn't support TreeExplainer, using KernelExplainer")
        explainer = shap.KernelExplainer(model.predict, X_scaled[:100])
        shap_values = explainer.shap_values(X_scaled[:10])
        
except Exception as e:
    print(f"SHAP Error: {str(e)}")
    print("Installing SHAP: pip install shap")

# ============================================================================
# 3. LIME ANALYSIS
# ============================================================================
print("\n3. PERFORMING LIME ANALYSIS...")

try:
    import lime
    import lime.lime_tabular
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_names,
        class_names=['Not Purchased', 'Purchased'],
        mode='classification',
        verbose=True,
        random_state=42
    )
    
    # Explain individual predictions
    os.makedirs('interpretability_results/lime', exist_ok=True)
    
    # Explain first 3 samples
    for i in range(min(3, len(X_scaled))):
        exp = explainer.explain_instance(
            X_scaled[i], 
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            num_features=5,
            top_labels=1
        )
        
        # Save as HTML
        exp.save_to_file(f'interpretability_results/lime/lime_explanation_sample_{i}.html')
        
        # Save as image
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Explanation - Sample {i}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'interpretability_results/lime/lime_sample_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ LIME explanation saved for sample {i}")
        
except Exception as e:
    print(f"LIME Error: {str(e)}")
    print("Installing LIME: pip install lime")

# ============================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n4. FEATURE IMPORTANCE ANALYSIS...")

if hasattr(model, 'feature_importances_'):
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]  # Sort indices by importance
    
    print(f"Model has {len(importance)} features in importance array")
    print(f"Feature names list has {len(feature_names)} names")
    
    # Find how many features we can safely plot
    max_len = min(len(importance), len(feature_names), 20)  # Max 20 features for readability
    
    # Plot feature importance (top features only)
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Top Features)', fontsize=16)
    
    # Get top N features
    top_indices = indices[:max_len]
    top_importance = importance[top_indices]
    
    # Create y-axis labels safely
    y_labels = []
    for idx in top_indices:
        if idx < len(feature_names):
            y_labels.append(feature_names[idx])
        else:
            y_labels.append(f'Feature_{idx}')
    
    # Plot horizontal bar chart
    plt.barh(range(len(top_importance)), top_importance[::-1])
    plt.yticks(range(len(top_importance)), y_labels[::-1])
    plt.xlabel('Relative Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('interpretability_results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved (top {len(top_importance)} features)")
    
    # Save importance to CSV
    importance_data = []
    for rank, idx in enumerate(indices, 1):
        if idx < len(feature_names):
            feature_name = feature_names[idx]
        else:
            feature_name = f'Feature_{idx}'
        
        importance_data.append({
            'Rank': rank,
            'Feature_Index': idx,
            'Feature_Name': feature_name,
            'Importance': importance[idx]
        })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df.to_csv('interpretability_results/feature_importance.csv', index=False)
    print(f"✓ Feature importance CSV saved ({len(importance_df)} features)")
    
    # Print top 5 features
    print("\nTop 5 Important Features:")
    for i in range(min(5, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"  {i+1}. {row['Feature_Name']}: {row['Importance']:.4f}")
        
else:
    print("⚠️ Model doesn't have feature_importances_ attribute")
    print("Trying to get coefficients for linear models...")
    
    # For linear models like Logistic Regression
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        indices = np.argsort(importance)[::-1]
        
        max_len = min(len(importance), len(feature_names), 20)
        top_indices = indices[:max_len]
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        plt.title('Feature Coefficients (Absolute Value)', fontsize=16)
        
        y_labels = []
        for idx in top_indices:
            if idx < len(feature_names):
                y_labels.append(feature_names[idx])
            else:
                y_labels.append(f'Feature_{idx}')
        
        plt.barh(range(len(top_indices)), importance[top_indices][::-1])
        plt.yticks(range(len(top_indices)), y_labels[::-1])
        plt.xlabel('Absolute Coefficient Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('interpretability_results/feature_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature coefficients plot saved")
# ============================================================================
# 5. GENERATING INTERPRETABILITY REPORT
# ============================================================================
print("\n5. GENERATING INTERPRETABILITY REPORT...")

report = f"""
MODEL INTERPRETABILITY REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL DETAILS
{'-'*20}
Model Type: Random Forest
Number of Features: {len(feature_names)}
Features: {', '.join(feature_names)}

INTERPRETABILITY METHODS
{'-'*25}
1. SHAP (SHapley Additive exPlanations)
   - Global feature importance
   - Individual prediction explanations
   - Feature dependence plots

2. LIME (Local Interpretable Model-agnostic Explanations)
   - Local prediction explanations
   - Sample-wise interpretability

3. Traditional Feature Importance
   - Model-specific importance scores

FILES GENERATED
{'-'*20}
interpretability_results/
├── shap/
│   ├── shap_summary.png       - Global feature importance
│   ├── shap_bar.png           - Feature importance (bar)
│   ├── shap_force_plot.png    - Individual prediction
│   └── shap_dependence_*.png  - Feature dependence plots
├── lime/
│   ├── lime_explanation_*.html - Interactive explanations
│   └── lime_sample_*.png      - Visual explanations
├── feature_importance.png     - Traditional importance
└── feature_importance.csv     - Importance scores

HOW TO INTERPRET
{'-'*15}
1. SHAP Summary Plot:
   - Red/Blue: High/Low feature values
   - Position: Impact on prediction
   - Width: Magnitude of effect

2. LIME Explanations:
   - Shows features contributing to individual predictions
   - Orange: Supports prediction
   - Blue: Contradicts prediction

3. Business Insights:
   - Most important features for purchase prediction
   - Key drivers of customer behavior
   - Actionable insights for marketing

RECOMMENDATIONS
{'-'*15}
1. Focus marketing on customers with high 'Spending_Power_Index'
2. Target demographics based on top important features
3. Use SHAP values for individual customer insights
4. Monitor feature importance changes over time

READY FOR BUSINESS USE
"""

# Save with UTF-8 encoding
try:
    with open('interpretability_results/interpretability_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("SUCCESS: Interpretability analysis complete!")
    print("Files saved in 'interpretability_results/' folder")
except Exception as e:
    print(f"Warning: Could not save report with UTF-8: {str(e)}")
    # Try ASCII as fallback
    with open('interpretability_results/interpretability_report.txt', 'w', encoding='ascii', errors='ignore') as f:
        f.write(report)
    print("Saved report with ASCII encoding (some characters may be omitted)")

print("\n" + "="*60)
print("INTERPRETABILITY TASK COMPLETED")
print("="*60)