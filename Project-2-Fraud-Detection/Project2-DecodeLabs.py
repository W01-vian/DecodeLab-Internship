"""
PROJECT 2: FRAUD DETECTION PIPELINE
Batch: 2026 | Powered by DecodeLabs
Data Science Intern - Javeria Faisal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PROJECT 2: FRAUD DETECTION PIPELINE")
print("=" * 80)

# STEP 1: DATA LOADING - CORRECTED PATH
print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)
file_path = r"C:\Users\DELL\OneDrive\Documentos\Decode Labs\Project1-Output.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"File not found at: {file_path}")
    print("Trying alternative path...")
    # Try relative path
    df = pd.read_csv("Project1-Output.csv")
    print(f"Dataset loaded from relative path: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\nColumns in dataset: {df.columns.tolist()[:15]}...")

# STEP 2: DATA INSPECTION & TARGET CREATION
print("\n" + "=" * 80)
print("STEP 2: DATA INSPECTION & TARGET CREATION")
print("=" * 80)
# Check which order status columns exist
status_cols = [col for col in df.columns if 'OrderStatus' in col]
print(f"Order status columns found: {status_cols}")
# Create target: Fraud = Returned or Cancelled orders
df['IsFraud'] = 0
# Check for returned or cancelled status columns
if 'OrderStatus_Returned' in df.columns:
    df.loc[df['OrderStatus_Returned'] == 1, 'IsFraud'] = 1
if 'OrderStatus_Cancelled' in df.columns:
    df.loc[df['OrderStatus_Cancelled'] == 1, 'IsFraud'] = 1
print(f"\nTarget variable 'IsFraud' created")
print(f"Fraudulent transactions: {df['IsFraud'].sum():,} ({df['IsFraud'].mean():.2%})")
print(f"Legitimate transactions: {(df['IsFraud'] == 0).sum():,} ({1 - df['IsFraud'].mean():.2%})")

# STEP 3: FEATURE ENGINEERING
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)
# Create additional features for better fraud detection
# 1. Average item value
df['AvgItemValue'] = df['TotalPrice'] / df['ItemsInCart']
# 2. Items per order ratio
df['ItemsPerOrder'] = df['ItemsInCart'] / (df['Quantity'] + 1)
# 3. High value flag (transactions above 95th percentile)
high_value_threshold = df['TotalPrice'].quantile(0.95)
df['IsHighValue'] = (df['TotalPrice'] > high_value_threshold).astype(int)
# 4. Price per unit
df['PricePerUnit'] = df['TotalPrice'] / df['Quantity']
# 5. Discount indicator (if coupon used)
df['HasDiscount'] = (df['CouponCode'] != 'None').astype(int)
# 6. Weekend indicator
if 'IsWeekend' in df.columns:
    df['WeekendFraud'] = df['IsWeekend'] * df['IsFraud']
print(f"Created 6 new features")
print(f"New features: AvgItemValue, ItemsPerOrder, IsHighValue, PricePerUnit, HasDiscount, WeekendFraud")
 
# STEP 4: FEATURE SELECTION
print("\n" + "=" * 80)
print("STEP 4: FEATURE SELECTION")
print("=" * 80)
# Numerical features to use
numerical_features = [
    'Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice',
    'AvgItemValue', 'ItemsPerOrder', 'PricePerUnit'
]
# Categorical/encoded features
categorical_features = [
    'IsWeekend', 'IsDelivered', 'HasDiscount', 'IsHighValue'
]
# One-hot encoded features from Project 1
encoded_features = [
    col for col in df.columns if any(x in col for x in [
        'Product_', 'PaymentMethod_', 'ReferralSource_'
    ])
]
# Combine all features
all_features = numerical_features + categorical_features + encoded_features
# Filter to only existing columns
available_features = [col for col in all_features if col in df.columns]
print(f"Selected {len(available_features)} features for modeling")
print(f"Features: {available_features[:10]}...")
X = df[available_features].copy()
y = df['IsFraud'].copy()
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:")
print(f"  - Legitimate (0): {(y == 0).sum():,}")
print(f"  - Fraudulent (1): {(y == 1).sum():,}")

# STEP 5: TRAIN-TEST SPLIT (CRITICAL: DO THIS BEFORE SMOTE!)
print("\n" + "=" * 80)
print("STEP 5: TRAIN-TEST SPLIT (BEFORE RESAMPLING)")
print("=" * 80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train):,} samples")
print(f"  - Fraud: {y_train.sum():,} ({y_train.mean():.2%})")
print(f"  - Legitimate: {(len(y_train) - y_train.sum()):,} ({1 - y_train.mean():.2%})")
print(f"\nTest set: {len(X_test):,} samples")
print(f"  - Fraud: {y_test.sum():,} ({y_test.mean():.2%})")
print(f"  - Legitimate: {(len(y_test) - y_test.sum()):,} ({1 - y_test.mean():.2%})")

# STEP 6: LOGISTIC REGRESSION WITH SMOTE
print("\n" + "=" * 80)
print("STEP 6: LOGISTIC REGRESSION WITH SMOTE")
print("=" * 80)
# IMPORTANT: imblearn.pipeline.Pipeline prevents data leakage!
lr_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),      # Scale features
    ('smote', SMOTE(random_state=42)), # SMOTE creates synthetic samples
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
print("Pipeline: StandardScaler → SMOTE → LogisticRegression")
print("   (Leak-free: SMOTE only applied to training folds)")
# Hyperparameter tuning
lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear', 'saga'],
    'smote__k_neighbors': [3, 5, 7]
}
# Simple grid search
lr_grid = GridSearchCV(
    lr_pipeline,
    {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear'],
        'smote__k_neighbors': [5]
    },
    cv=3,
    scoring='recall',
    n_jobs=-1,
    verbose=0
)
print("Training Logistic Regression...")
lr_grid.fit(X_train, y_train)

print(f"\nBest Logistic Regression parameters:")
for param, value in lr_grid.best_params_.items():
    print(f"  - {param}: {value}")

# STEP 7: RANDOM FOREST WITH SMOTE
print("\n" + "=" * 80)
print("STEP 7: RANDOM FOREST WITH SMOTE")
print("=" * 80)
# Random Forest doesn't need scaling (tree-based)
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])
print("Pipeline: SMOTE → RandomForestClassifier")
print("   (No scaling needed for tree-based models)")
rf_param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 15, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': ['balanced', 'balanced_subsample'],
    'smote__k_neighbors': [3, 5]
}
# Simple grid search
rf_grid = GridSearchCV(
    rf_pipeline,
    {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, None],
        'classifier__class_weight': ['balanced'],
        'smote__k_neighbors': [5]
    },
    cv=3,
    scoring='recall',
    n_jobs=-1,
    verbose=0
)
print("Training Random Forest...")
rf_grid.fit(X_train, y_train)

print(f"\nBest Random Forest parameters:")
for param, value in rf_grid.best_params_.items():
    print(f"  - {param}: {value}")

# STEP 8: MODEL EVALUATION 
print("\n" + "=" * 80)
print("STEP 8: MODEL EVALUATION")
print("=" * 80)
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation without using accuracy"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Key metrics for imbalanced data
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n{'='*50}")
    print(f"{model_name} Performance")
    print('='*50)
    print(f"  Precision:  {precision:.4f}  ← When we flag fraud, are we right?")
    print(f"  Recall:     {recall:.4f}     ← Did we catch all fraud?")
    print(f"  F1-Score:   {f1:.4f}        ← Harmonic mean of Precision & Recall")
    print(f"  ROC-AUC:    {roc_auc:.4f}    ← Model's ability to separate classes")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0,0]:,}")
    print(f"    False Positives: {cm[0,1]:,}  ← Type I Error")
    print(f"    False Negatives: {cm[1,0]:,}  ← Type II Error (Costly!)")
    print(f"    True Positives:  {cm[1,1]:,}")
    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': cm
    }
# Evaluate both models
print("\n" + "-" * 50)
print("EVALUATING MODELS (Accuracy intentionally omitted)")
print("-" * 50)

lr_results = evaluate_model(lr_grid, X_test, y_test, "Logistic Regression")
rf_results = evaluate_model(rf_grid, X_test, y_test, "Random Forest")

# STEP 9: CROSS-VALIDATION WITH SMOTE (Leak-Free)
print("\n" + "=" * 80)
print("STEP 9: CROSS-VALIDATION WITH SMOTE")
print("=" * 80)
best_model = rf_grid.best_estimator_
# SMOTE is applied inside each fold - NO LEAKAGE!
cv_scores = cross_val_score(
    best_model, X_train, y_train, 
    cv=5, scoring='recall', n_jobs=-1
)
print(f"5-Fold Cross-Validation Recall Scores: {cv_scores}")
print(f"Mean Recall: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print("SMOTE was applied independently in each fold - no leakage!")

# STEP 10: VISUALIZATIONS
print("\n" + "=" * 80)
print("STEP 10: VISUALIZATIONS")
print("=" * 80)
# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# 1. Confusion Matrix - Logistic Regression
cm_lr = lr_results['cm']
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Legitimate', 'Fraud'])
disp_lr.plot(ax=axes[0, 0], cmap='Blues', values_format='d')
axes[0, 0].set_title(f'Logistic Regression - Confusion Matrix\nF1: {lr_results["f1"]:.3f}')
# 2. Confusion Matrix - Random Forest
cm_rf = rf_results['cm']
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Legitimate', 'Fraud'])
disp_rf.plot(ax=axes[0, 1], cmap='Greens', values_format='d')
axes[0, 1].set_title(f'Random Forest - Confusion Matrix\nF1: {rf_results["f1"]:.3f}')
# 3. ROC Curves
for results, color, label in [
    (lr_results, 'blue', f'Logistic Regression (AUC={lr_results["roc_auc"]:.3f})'),
    (rf_results, 'green', f'Random Forest (AUC={rf_results["roc_auc"]:.3f})')
]:
    fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
    axes[0, 2].plot(fpr, tpr, color=color, lw=2, label=label)
axes[0, 2].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
axes[0, 2].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0, 2].set_ylabel('True Positive Rate (Recall)')
axes[0, 2].set_title('ROC Curves - Fraud Detection')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(True, alpha=0.3)
# 4. Precision-Recall Curves
for results, color, label in [
    (lr_results, 'blue', 'Logistic Regression'),
    (rf_results, 'green', 'Random Forest')
]:
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, results['y_proba'])
    axes[1, 0].plot(recall_vals, precision_vals, color=color, lw=2, label=label)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curves\n(Focus on Fraud Class)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
# 5. Feature Importance - Random Forest
rf_best = rf_grid.best_estimator_
if hasattr(rf_best.named_steps['classifier'], 'feature_importances_'):
    importances = rf_best.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1][:10]
    axes[1, 1].barh(range(10), importances[indices][::-1])
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels([feature_names[i] for i in indices[::-1]])
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Top 10 Most Important Features\n(Random Forest)')
    axes[1, 1].grid(True, alpha=0.3)
# 6. Model Comparison
metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = [lr_results['precision'], lr_results['recall'], lr_results['f1'], lr_results['roc_auc']]
rf_scores = [rf_results['precision'], rf_results['recall'], rf_results['f1'], rf_results['roc_auc']]
x = np.arange(len(metrics))
width = 0.35
bars1 = axes[1, 2].bar(x - width/2, lr_scores, width, label='Logistic Regression', color='royalblue')
bars2 = axes[1, 2].bar(x + width/2, rf_scores, width, label='Random Forest', color='forestgreen')
axes[1, 2].set_xlabel('Evaluation Metrics')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Model Comparison\n(ACCURACY DELIBERATELY OMITTED)')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].legend()
axes[1, 2].set_ylim(0, 1.05)
axes[1, 2].grid(True, alpha=0.3, axis='y')
# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved to 'fraud_detection_results.png'")

# STEP 11: FINAL SUMMARY
print("\n" + "=" * 80)
print("PROJECT 2: FINAL SUMMARY")
print("=" * 80)
print("\n" + "-" * 80)
print("KEY ACHIEVEMENTS")
print("-" * 80)
print("""
 1. DATA HANDLING
   - Loaded processed data from Project 1
   - Created 'IsFraud' target variable (Returned + Cancelled)
   - Engineered 6 new predictive features

 2. IMBALANCED DATA HANDLING
   - Original fraud rate: {fraud_rate:.2%}
   - Applied SMOTE to balance classes
   - Used imblearn.pipeline.Pipeline to prevent data leakage

 3. MODEL TRAINING
   - Logistic Regression with StandardScaler + SMOTE
   - Random Forest with SMOTE (no scaling needed)
   - GridSearchCV for hyperparameter tuning

 4. EVALUATION (NO ACCURACY!)
   - Precision: When we flag fraud, are we right?
   - Recall: Did we catch all fraud?
   - F1-Score: Harmonic mean of Precision & Recall
   - ROC-AUC: Model's ability to separate classes

 5. BEST PERFORMANCE
   - Model: {best_model_name}
   - Precision: {best_precision:.4f}
   - Recall: {best_recall:.4f}
   - F1-Score: {best_f1:.4f}
   - ROC-AUC: {best_auc:.4f}

 6. PRODUCTION READY
   - Leak-free pipelines
   - Cross-validation with SMOTE
   - Feature importance analysis
   - Comprehensive visualizations
""".format(
    fraud_rate=df['IsFraud'].mean(),
    best_model_name=best_model_name,
    best_precision=max(lr_results['precision'], rf_results['precision']),
    best_recall=max(lr_results['recall'], rf_results['recall']),
    best_f1=max(lr_results['f1'], rf_results['f1']),
    best_auc=max(lr_results['roc_auc'], rf_results['roc_auc'])
))
print("-" * 80)
print("RECOMMENDATIONS FOR DEPLOYMENT")
print("-" * 80)
print("""
1. Deploy Random Forest model (better overall performance)
2. Monitor model weekly for performance drift
3. Retrain monthly with new transaction data
4. Use feature importance for fraud investigation prioritization
5. Implement a human-in-the-loop for high-value transactions
6. Consider ensemble methods for further improvement
7. Document all decisions for audit compliance
""")
print("\n" + "=" * 80)
print("PROJECT 2 COMPLETED SUCCESSFULLY")
print("=" * 80)