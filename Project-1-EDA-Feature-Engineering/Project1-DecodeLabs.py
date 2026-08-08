"""
PROJECT 1: ADVANCED EDA & FEATURE ENGINEERING
Batch: 2026 | Powered by DecodeLabs
Data Science Intern - Javeria Faisal
Requirements:
1. Handle missing data via statistical imputation (Mean/Median/KNN)
2. Identify and neutralize outliers using IQR/Z-Scores
3. Engineer at least 3 new predictive features
4. Use Pandas, NumPy, statistical analysis, data cleaning, feature extraction
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
print("=" * 80)
print("PROJECT 1: ADVANCED EDA & FEATURE ENGINEERING")
print("=" * 80)

# 1. DATA LOADING & INITIAL INSPECTION
file_path = r"C:\Users\DELL\OneDrive\Documentos\Decode Labs\Dataset for Data Analytics.xlsx"
df = pd.read_excel(file_path)
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nDescriptive Statistics:")
print(df.describe())

# 2. CHECK FOR MISSING VALUES
print("\n" + "=" * 80)
print("PHASE 1: HANDLING MISSING DATA")
print("=" * 80)
missing_data = df.isnull().sum()
print(f"\nMissing values per column:\n{missing_data[missing_data > 0]}")
df_clean = df.copy()

# 3. HANDLE MISSING DATA - CATEGORICAL
df_clean['CouponCode'] = df_clean['CouponCode'].fillna('None')
print("\nFilled missing 'CouponCode' with 'None'")

# 4. HANDLE MISSING DATA - NUMERICAL (Statistical Imputation)
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
missing_numerical = df_clean[numerical_cols].isnull().sum()
if missing_numerical.sum() > 0:
    print(f"\nNumerical columns with missing values:\n{missing_numerical[missing_numerical > 0]}")
    for col in missing_numerical[missing_numerical > 0].index:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"Imputed '{col}' with median: {median_val:.2f}")
else:
    print("\nNo missing values in numerical columns")

# 5. KNN IMPUTATION DEMONSTRATION
print("\nKNN Imputation Demonstration:")
df_knn_demo = df_clean.copy()
df_knn_demo.loc[0:2, 'UnitPrice'] = np.nan
df_knn_demo.loc[5:7, 'Quantity'] = np.nan
knn_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']
knn_data = df_knn_demo[knn_cols]
scaler = StandardScaler()
knn_scaled = scaler.fit_transform(knn_data)
imputer = KNNImputer(n_neighbors=5)
knn_imputed_scaled = imputer.fit_transform(knn_scaled)
knn_imputed = scaler.inverse_transform(knn_imputed_scaled)
for i, col in enumerate(knn_cols):
    df_knn_demo[col] = knn_imputed[:, i]
print("KNN imputation demonstrated on sample data")

# 6. OUTLIER DETECTION & NEUTRALIZATION (IQR Method)
print("\n" + "=" * 80)
print("PHASE 2: OUTLIER DETECTION & NEUTRALIZATION")
print("=" * 80)
outlier_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart']
outlier_report = []
for col in outlier_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_report.append({
        'Column': col,
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outliers Found': outlier_count
    })
    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    if outlier_count > 0:
        print(f"Capped {outlier_count} outliers in '{col}'")
    else:
        print(f"No outliers found in '{col}'")
outlier_df = pd.DataFrame(outlier_report)
print("\nOutlier Report:")
print(outlier_df.to_string(index=False))

# 7. FEATURE ENGINEERING
print("\n" + "=" * 80)
print("PHASE 3: FEATURE ENGINEERING")
print("=" * 80)
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
# Feature 1: Day of Week
df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
print("Created Feature 1: 'DayOfWeek'")
# Feature 2: Is Weekend
df_clean['IsWeekend'] = (df_clean['Date'].dt.dayofweek >= 5).astype(int)
print("Created Feature 2: 'IsWeekend'")
# Feature 3: Order Month
df_clean['OrderMonth'] = df_clean['Date'].dt.month
print("Created Feature 3: 'OrderMonth'")
# Feature 4: Is Delivered
df_clean['IsDelivered'] = (df_clean['OrderStatus'] == 'Delivered').astype(int)
print("Created Feature 4: 'IsDelivered'")
# Feature 5: Has Discount
df_clean['HasDiscount'] = (df_clean['CouponCode'] != 'None').astype(int)
print("Created Feature 5: 'HasDiscount'")
# Feature 6: Average Item Value
df_clean['AvgItemValue'] = df_clean['TotalPrice'] / df_clean['ItemsInCart']
print("Created Feature 6: 'AvgItemValue'")
# Feature 7: Order Year
df_clean['OrderYear'] = df_clean['Date'].dt.year
print("Created Feature 7: 'OrderYear'")
# Feature 8: Discount Code Category
df_clean['DiscountCode'] = df_clean['CouponCode'].apply(
    lambda x: 'None' if x == 'None' else 
              ('SAVE10' if 'SAVE' in str(x) else 
               ('WINTER15' if 'WINTER' in str(x) else 
                ('FREESHIP' if 'FREE' in str(x) else 'Other')))
)
print("Created Feature 8: 'DiscountCode'")
print(f"\nTotal features created: 8 new features")

# 8. CATEGORICAL ENCODING
print("\n" + "=" * 80)
print("PHASE 4: CATEGORICAL ENCODING")
print("=" * 80)
categorical_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'ReferralSource', 
                    'DayOfWeek', 'DiscountCode']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"One-Hot Encoding applied to {len(categorical_cols)} categorical columns")
print(f"New shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")

# 9. MULTICOLLINEARITY CHECK
print("\n" + "=" * 80)
print("PHASE 5: MULTICOLLINEARITY CHECK")
print("=" * 80)
numerical_cols_encoded = df_encoded.select_dtypes(include=[np.number]).columns
corr_matrix = df_encoded[numerical_cols_encoded].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
if high_corr_pairs:
    print(f"Found {len(high_corr_pairs)} highly correlated pairs:")
    for pair in high_corr_pairs[:5]:
        print(f"   - {pair['Feature 1']} vs {pair['Feature 2']}: {pair['Correlation']:.3f}")
    to_drop = set()
    for pair in high_corr_pairs:
        if pair['Feature 1'] in df_encoded.columns and pair['Feature 2'] in df_encoded.columns:
            if df_encoded[pair['Feature 1']].var() < df_encoded[pair['Feature 2']].var():
                to_drop.add(pair['Feature 1'])
            else:
                to_drop.add(pair['Feature 2'])
    to_drop = [col for col in to_drop if col in df_encoded.columns]
    if to_drop:
        df_encoded = df_encoded.drop(columns=to_drop, errors='ignore')
        print(f"Dropped {len(to_drop)} features to reduce multicollinearity")
else:
    print("No high multicollinearity detected")

# 10. FEATURE SCALING
print("\n" + "=" * 80)
print("PHASE 6: FEATURE SCALING")
print("=" * 80)
scale_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart', 
              'AvgItemValue', 'OrderMonth', 'OrderYear']

binary_cols = ['IsWeekend', 'IsDelivered', 'HasDiscount']
scale_cols = [col for col in scale_cols if col in df_encoded.columns and col not in binary_cols]
if scale_cols:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded[scale_cols])
    for i, col in enumerate(scale_cols):
        df_encoded[f'{col}_scaled'] = scaled_data[:, i]
    print(f"Scaled {len(scale_cols)} numerical features")
else:
    print("No features to scale")

# 11. FINAL DATA OVERVIEW
print("\n" + "=" * 80)
print("FINAL DATA SUMMARY")
print("=" * 80)
print(f"\nFinal Dataset Shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
print(f"Missing Values: {df_encoded.isnull().sum().sum()}")
print(f"Memory Usage: {df_encoded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nFirst 5 rows of processed data:")
print(df_encoded.head())
print("\nColumn Types:")
print(f"   - Numerical: {len(df_encoded.select_dtypes(include=[np.number]).columns)}")
print(f"   - Categorical: {len(df_encoded.select_dtypes(include=['object']).columns)}")

# 12. EDA SUMMARY
print("\n" + "=" * 80)
print("EDA SUMMARY - KEY INSIGHTS")
print("=" * 80)
print("\nNumerical Columns Statistics:")
stats_df = pd.DataFrame({
    'Column': ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart'],
    'Mean': [df_clean['Quantity'].mean(), df_clean['UnitPrice'].mean(), 
             df_clean['TotalPrice'].mean(), df_clean['ItemsInCart'].mean()],
    'Std': [df_clean['Quantity'].std(), df_clean['UnitPrice'].std(), 
            df_clean['TotalPrice'].std(), df_clean['ItemsInCart'].std()],
    'Skew': [df_clean['Quantity'].skew(), df_clean['UnitPrice'].skew(), 
             df_clean['TotalPrice'].skew(), df_clean['ItemsInCart'].skew()]
})
print(stats_df.round(2).to_string(index=False))
print("\nTop Products by Order Frequency:")
product_counts = df_clean['Product'].value_counts().head(5)
for product, count in product_counts.items():
    print(f"   - {product}: {count} orders ({count/len(df_clean)*100:.1f}%)")
print("\nPayment Method Distribution:")
payment_counts = df_clean['PaymentMethod'].value_counts().head(5)
for method, count in payment_counts.items():
    print(f"   - {method}: {count} orders ({count/len(df_clean)*100:.1f}%)")
print("\nOrder Status Distribution:")
status_counts = df_clean['OrderStatus'].value_counts()
for status, count in status_counts.items():
    print(f"   - {status}: {count} orders ({count/len(df_clean)*100:.1f}%)")
print("\nWeekend vs Weekday Orders:")
weekend_count = df_clean['IsWeekend'].sum()
total_orders = len(df_clean)
print(f"   - Weekend Orders: {weekend_count} ({weekend_count/total_orders*100:.1f}%)")
print(f"   - Weekday Orders: {total_orders - weekend_count} ({(total_orders - weekend_count)/total_orders*100:.1f}%)")
print("\nDiscount Usage:")
discount_count = df_clean['HasDiscount'].sum()
print(f"   - Orders with Discount: {discount_count} ({discount_count/total_orders*100:.1f}%)")
print(f"   - Orders without Discount: {total_orders - discount_count} ({(total_orders - discount_count)/total_orders*100:.1f}%)")

# 13. SAVE PROCESSED DATA
print("\n" + "=" * 80)
print("SAVING OUTPUT")
print("=" * 80)
import os
os.makedirs('output', exist_ok=True)
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'output/processed_data_{timestamp}.csv'
df_encoded.to_csv(output_file, index=False)
print(f"Processed data saved to: {output_file}")

# 14. SUMMARY
print("\n" + "=" * 80)
print("PROJECT 1 COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nWhat was accomplished:")
print("   1. Loaded and inspected dataset")
print("   2. Handled missing data with Mode/Median imputation")
print("   3. Demonstrated KNN imputation")
print("   4. Detected outliers using IQR method")
print("   5. Neutralized outliers using Winsorization (capping)")
print("   6. Engineered 8 new predictive features")
print("   7. Encoded categorical variables")
print("   8. Checked and mitigated multicollinearity")
print("   9. Scaled numerical features")
print("  10. Generated EDA insights and summary")
print("  11. Saved processed data for machine learning")
print("\nNext Steps:")
print("   - Use the processed data for ML modeling")
print("   - Compare different imputation methods")
print("   - Test different feature engineering techniques")
print("   - Build predictive models and evaluate performance")
print("\n" + "=" * 80)
