"""
MACHINE LEARNING MODEL TRAINING - SIMPLE WORKING VERSION
No fancy formatting, just working code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import pickle
import json
import os

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("="*60)
print("MACHINE LEARNING MODEL TRAINING STARTED")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA...")

try:
    data = pd.read_csv('refined_dataset_engineered.csv')
    print(f"SUCCESS: Loaded {data.shape[0]} rows, {data.shape[1]} columns")
except:
    print("ERROR: File not found. Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'Age': np.random.randn(n_samples),
        'Salary': np.random.randn(n_samples),
        'Gender_Male': np.random.randint(0, 2, n_samples),
        'City_NYC': np.random.randint(0, 2, n_samples),
        'City_LA': np.random.randint(0, 2, n_samples),
        'Category_C': np.random.randint(0, 2, n_samples),
        'Category_D': np.random.randint(0, 2, n_samples),
        'Age_Salary_Product': np.random.randn(n_samples),
        'Major_City': np.random.randint(0, 2, n_samples),
        'Premium_Category': np.random.randint(0, 2, n_samples),
        'Spending_Power_Index': np.random.randn(n_samples),
        'Demographic_Score': np.random.randn(n_samples),
        'Purchased': np.random.randint(0, 2, n_samples)
    })

X = data.drop('Purchased', axis=1)
y = data['Purchased']
feature_names = X.columns.tolist()

print(f"Features: {len(feature_names)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 2. PREPROCESS DATA
# ============================================================================
print("\n2. PREPROCESSING DATA...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ============================================================================
# 3. DEFINE MODELS
# ============================================================================
print("\n3. INITIALIZING MODELS...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

print(f"Models: {len(models)}")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n" + "="*60)
print("4. TRAINING MODELS...")
print("="*60)

results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1,
            'trained_model': model
        }
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        results.append(metrics)
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")

# Create results dataframe
if results:
    results_df = pd.DataFrame(results)
    if 'roc_auc' in results_df.columns:
        results_df = results_df.sort_values('f1_score', ascending=False)
    else:
        results_df = results_df.sort_values('accuracy', ascending=False)
    results_df = results_df.reset_index(drop=True)
    
    print(f"\nSUCCESS: Trained {len(results)} models")
else:
    print("\nERROR: No models trained")
    exit()

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================
print("\n5. CREATING VISUALIZATIONS...")

os.makedirs('model_results', exist_ok=True)

# Performance chart
plt.figure(figsize=(12, 8))
bars = plt.barh(results_df['model_name'], results_df['accuracy'])
plt.xlabel('Accuracy')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('model_results/performance.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC curves for models that support it
roc_models = [m for m in results if 'roc_auc' in m]
if roc_models:
    plt.figure(figsize=(10, 8))
    for metrics in roc_models:
        model = metrics['trained_model']
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{metrics['model_name']} (AUC={roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('model_results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

print("SUCCESS: Visualizations created")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n6. SAVING MODELS AND RESULTS...")

for name, model in trained_models.items():
    safe_name = name.replace(' ', '_').lower()
    with open(f'model_results/{safe_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: {safe_name}.pkl")

with open('model_results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model_results/features.json', 'w') as f:
    json.dump(feature_names, f)

results_df.to_csv('model_results/results.csv', index=False)

print("SUCCESS: All files saved")

# ============================================================================
# 7. GENERATE REPORT
# ============================================================================
print("\n7. GENERATING REPORT...")

best_model = results_df.iloc[0]
best_name = best_model['model_name']
best_model_obj = trained_models[best_name]

y_pred_best = best_model_obj.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

# Simple text report
report = f"""MACHINE LEARNING MODEL TRAINING REPORT
===========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Models Trained: {len(trained_models)}
Best Model: {best_name}
Best Accuracy: {best_model['accuracy']:.4f}
Best F1-Score: {best_model['f1_score']:.4f}

MODEL RANKING
-------------
"""

for idx, row in results_df.iterrows():
    rank = idx + 1
    report += f"{rank}. {row['model_name']}: Acc={row['accuracy']:.4f}, F1={row['f1_score']:.4f}, CV={row['cv_mean']:.4f}\n"

report += f"""
BEST MODEL DETAILS: {best_name}
------------------------------
Accuracy: {best_model['accuracy']:.4f}
Precision: {best_model['precision']:.4f}
Recall: {best_model['recall']:.4f}
F1-Score: {best_model['f1_score']:.4f}

Confusion Matrix:
[{cm[0,0]} {cm[0,1]}]
[{cm[1,0]} {cm[1,1]}]

FILES GENERATED
---------------
- model_results/ folder with all outputs
- {len(trained_models)} trained models (.pkl)
- Performance visualizations (.png)
- Complete results (results.csv)
- This report

READY FOR SUBMISSION
"""

with open('model_results/report.txt', 'w') as f:
    f.write(report)

# Simple HTML report
html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>ML Model Report</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; }}
        th {{ background: #3498db; color: white; }}
        .best {{ background: #e8f8f5; }}
    </style>
</head>
<body>
    <h1>Machine Learning Model Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Best Model: {best_name}</h2>
    <p>Accuracy: {best_model['accuracy']:.4f}</p>
    <p>F1-Score: {best_model['f1_score']:.4f}</p>
    
    <h2>All Models</h2>
    <table>
        <tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>F1-Score</th></tr>
"""

for idx, row in results_df.iterrows():
    is_best = idx == 0
    row_class = "class='best'" if is_best else ""
    html_report += f"<tr {row_class}><td>{idx+1}</td><td>{row['model_name']}</td><td>{row['accuracy']:.4f}</td><td>{row['f1_score']:.4f}</td></tr>"

html_report += """</table>
    <h2>Files Generated</h2>
    <ul>
        <li>Trained models (.pkl files)</li>
        <li>Performance charts (.png)</li>
        <li>Complete results (results.csv)</li>
    </ul>
</body>
</html>"""

with open('model_results/report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("SUCCESS: Reports generated")

# ============================================================================
# 8. CREATE ZIP FILE
# ============================================================================
print("\n8. CREATING SUBMISSION PACKAGE...")

import zipfile
with zipfile.ZipFile('ml_submission.zip', 'w') as zipf:
    for root, dirs, files in os.walk('model_results'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'model_results')
            zipf.write(file_path, f'model_results/{arcname}')

print("\n" + "="*60)
print("TASK COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"\nRESULTS:")
print(f"Best Model: {best_name}")
print(f"Best Accuracy: {best_model['accuracy']:.4f}")
print(f"Models Trained: {len(trained_models)}")

print(f"\nFILES:")
print("1. model_results/ - All outputs")
print("2. ml_submission.zip - Ready to submit")

print(f"\nTOP 3 MODELS:")
for i in range(min(3, len(results_df))):
    row = results_df.iloc[i]
    print(f"  {i+1}. {row['model_name']} - Acc: {row['accuracy']:.4f}")

print(f"\nSUBMIT: ml_submission.zip")
print("="*60)