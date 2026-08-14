"""
PROJECT 3: CUSTOMER SEGMENTATION (UNSUPERVISED LEARNING)
Batch: 2026 | Powered by DecodeLabs
Data Science Intern - Javeria Faisal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("=" * 80)
print("PROJECT 3: CUSTOMER SEGMENTATION")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
print("\nSTEP 1: DATA LOADING")
print("-" * 40)

file_path = "Project1-Output.csv"
try:
    df_orders = pd.read_csv(file_path)
    print(f"Loaded {df_orders.shape[0]:,} orders, {df_orders.shape[1]} columns")
except FileNotFoundError:
    df_orders = pd.read_csv("C:/Users/DELL/OneDrive/Documentos/Decode Labs/Project1-Output.csv")
    print(f"Loaded from absolute path: {df_orders.shape[0]:,} orders")

# Convert Date to datetime
if 'Date' in df_orders.columns:
    df_orders['Date'] = pd.to_datetime(df_orders['Date'])
    print("Date column converted to datetime")

# ----------------------------------------------------------------------
# 2. CREATE PRODUCT COLUMN FROM ONE-HOT ENCODED COLUMNS
# ----------------------------------------------------------------------
print("\nSTEP 2: RECONSTRUCTING PRODUCT COLUMN")
print("-" * 40)

product_cols = [col for col in df_orders.columns if col.startswith('Product_')]
print(f"Found {len(product_cols)} product columns")

def get_product(row):
    for col in product_cols:
        if row[col] == 1:
            return col.replace('Product_', '')
    return 'Unknown'

df_orders['Product'] = df_orders.apply(get_product, axis=1)
print("Created 'Product' column from one-hot encoded columns")

# ----------------------------------------------------------------------
# 3. CUSTOMER AGGREGATION
# ----------------------------------------------------------------------
print("\nSTEP 3: CUSTOMER AGGREGATION")
print("-" * 40)

ref_date = df_orders['Date'].max() + pd.Timedelta(days=1)
print(f"Reference date for recency: {ref_date.strftime('%Y-%m-%d')}")

customer_df = df_orders.groupby('CustomerID').agg(
    Recency=('Date', lambda x: (ref_date - x.max()).days),
    Frequency=('OrderID', 'count'),
    Monetary=('TotalPrice', 'sum'),
    AvgOrderValue=('TotalPrice', 'mean'),
    AvgQuantity=('Quantity', 'mean'),
    AvgItemsInCart=('ItemsInCart', 'mean'),
    DiscountRate=('HasDiscount', 'mean'),
    DeliveryRate=('IsDelivered', 'mean'),
    UniqueProducts=('Product', lambda x: len(x.unique()))
).reset_index()

# Additional features
customer_df['MonetaryPerVisit'] = customer_df['Monetary'] / customer_df['Frequency']
customer_df['ItemsPerOrder'] = customer_df['AvgItemsInCart'] / customer_df['AvgQuantity']
customer_df['AvgPricePerItem'] = customer_df['AvgOrderValue'] / customer_df['AvgQuantity']

# Handle infinity values
customer_df = customer_df.replace([np.inf, -np.inf], np.nan)
customer_df = customer_df.fillna(0)

print(f"Created {customer_df.shape[0]:,} customer records with {customer_df.shape[1]-1} features")

# ----------------------------------------------------------------------
# 4. FEATURE SELECTION
# ----------------------------------------------------------------------
print("\nSTEP 4: FEATURE SELECTION")
print("-" * 40)

features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
            'AvgQuantity', 'AvgItemsInCart', 'DiscountRate', 'DeliveryRate',
            'MonetaryPerVisit', 'UniqueProducts', 'ItemsPerOrder', 'AvgPricePerItem']

# Handle missing values
missing = customer_df[features].isnull().sum()
if missing.sum() > 0:
    print(f"Missing values found: {missing[missing > 0]}")
    customer_df[features] = customer_df[features].fillna(customer_df[features].median())
    print("Missing values filled with median")

X = customer_df[features].copy()
print(f"Selected {len(features)} features for clustering")

# ----------------------------------------------------------------------
# 5. STANDARDIZE FEATURES
# ----------------------------------------------------------------------
print("\nSTEP 5: FEATURE STANDARDIZATION")
print("-" * 40)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized (mean=0, std=1)")

# ----------------------------------------------------------------------
# 6. PCA FOR DIMENSIONALITY REDUCTION
# ----------------------------------------------------------------------
print("\nSTEP 6: PRINCIPAL COMPONENT ANALYSIS")
print("-" * 40)

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Total features: {X_scaled.shape[1]}")
print(f"Components needed for 95% variance: {n_components_95}")

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance ratio per component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    cum_ratio = cumulative_variance[i]
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%) | Cumulative: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")

print(f"\nTotal explained variance: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")

# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print("2D PCA projection created for visualization")

# ----------------------------------------------------------------------
# 7. DETERMINE OPTIMAL K
# ----------------------------------------------------------------------
print("\nSTEP 7: DETERMINING OPTIMAL K")
print("-" * 40)

wcss = []
silhouette_scores = []
K_range = range(2, 11)

print("\nTesting K from 2 to 10:")
print("-" * 60)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"K={k:2d} | WCSS={kmeans.inertia_:,.2f} | Silhouette={score:.4f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K = {best_k} (highest Silhouette Score: {max(silhouette_scores):.4f})")

# ----------------------------------------------------------------------
# 8. K-MEANS CLUSTERING
# ----------------------------------------------------------------------
print("\nSTEP 8: K-MEANS CLUSTERING")
print("-" * 40)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
customer_df['Cluster'] = clusters

print(f"Clustering complete with K={best_k}")
print("\nCluster Distribution:")
cluster_counts = customer_df['Cluster'].value_counts().sort_index()
for c in cluster_counts.index:
    print(f"  Cluster {c}: {cluster_counts[c]:,} customers ({cluster_counts[c]/len(customer_df)*100:.1f}%)")

# ----------------------------------------------------------------------
# 9. REVERSE-ENGINEER CENTROIDS
# ----------------------------------------------------------------------
print("\nSTEP 9: CLUSTER CENTROIDS")
print("-" * 40)

centroids_scaled = pca.inverse_transform(kmeans.cluster_centers_)
centroids_original = scaler.inverse_transform(centroids_scaled)

# Create centroid DataFrame with ONLY features
centroid_df = pd.DataFrame(centroids_original, columns=features)
centroid_df.index.name = 'Cluster'
centroid_df.index = [f'Cluster {i}' for i in range(best_k)]

# Store size separately
centroid_df['Size'] = cluster_counts.values
centroid_df['Percentage'] = (cluster_counts.values / len(customer_df) * 100).round(1)

print("\nCluster Centroids (Original Units):")
print(centroid_df.round(2))

# ----------------------------------------------------------------------
# 10. CREATE BUSINESS PERSONAS (FIXED)
# ----------------------------------------------------------------------
print("\nSTEP 10: BUSINESS PERSONAS")
print("-" * 40)

# Calculate overall averages from original features
overall_avg = X.mean().values
persona_descriptions = []

for i in range(best_k):
    print(f"\n{'='*50}")
    print(f"CLUSTER {i} - {cluster_counts[i]:,} customers ({cluster_counts[i]/len(customer_df)*100:.1f}%)")
    print('='*50)
    
    # Get centroid values for features only (exclude Size and Percentage)
    centroid_features = centroid_df.iloc[i][features].values
    diff_pct = (centroid_features / overall_avg - 1) * 100
    
    top_features = np.argsort(np.abs(diff_pct))[::-1][:5]
    
    print("\nKey Distinguishing Traits:")
    for idx in top_features:
        feat = features[idx]
        val = centroid_features[idx]
        avg = overall_avg[idx]
        pct = diff_pct[idx]
        direction = "HIGHER" if pct > 0 else "LOWER"
        print(f"  {feat:20s}: {val:8.2f} ({abs(pct):5.1f}% {direction} than average)")
    
    # Persona classification
    if diff_pct[1] > 50 and diff_pct[2] > 50:
        persona = "VIP Customers"
        desc = "High frequency and high monetary value. Most valuable customers."
    elif diff_pct[0] > 30 and diff_pct[1] < -30:
        persona = "At-Risk Churners"
        desc = "Long time since last purchase. Need re-engagement campaigns."
    elif diff_pct[6] > 40:
        persona = "Deal Seekers"
        desc = "High discount usage. Price-sensitive but active."
    elif diff_pct[7] > 30 and diff_pct[1] > 20:
        persona = "Satisfied Regulars"
        desc = "High delivery rate, consistent purchases. Loyal and satisfied."
    elif diff_pct[9] > 50:
        persona = "Product Explorers"
        desc = "Buy diverse products. Cross-sell opportunities."
    elif diff_pct[0] < -20 and diff_pct[2] > 30:
        persona = "Recent Big Spenders"
        desc = "Very recent purchases with high spend. Hot leads."
    else:
        persona = "Average Shoppers"
        desc = "Balanced behavior across all metrics. Largest segment."
    
    persona_descriptions.append((i, persona, desc))
    print(f"\nPersona: {persona}")
    print(f"Description: {desc}")

# ----------------------------------------------------------------------
# 11. GENERATE VISUALIZATIONS
# ----------------------------------------------------------------------
print("\nSTEP 11: GENERATING VISUALIZATIONS")
print("-" * 40)

# Figure 1: Elbow + Silhouette
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
ax1.set_title('Elbow Method for Optimal K')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K = {best_k}')
ax1.legend()

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score for Optimal K')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Optimal K = {best_k}')
ax2.legend()
ax2.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('1_elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("Saved: 1_elbow_silhouette.png")
plt.close()

# Figure 2: 2D Cluster Visualization
fig2, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=clusters, cmap='viridis', alpha=0.6, s=80,
                     edgecolors='black', linewidth=0.5)

centroids_2d = pca_2d.transform(centroids_scaled)
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           marker='X', s=400, c='red', edgecolors='black',
           linewidth=3, label='Centroids', zorder=5)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title(f'Customer Clusters (K={best_k}) in 2D PCA Space')
ax.legend()
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Cluster', rotation=270, labelpad=25)

plt.tight_layout()
plt.savefig('2_clusters_2d.png', dpi=300, bbox_inches='tight')
print("Saved: 2_clusters_2d.png")
plt.close()

# Figure 3: Feature Distributions
fig3, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    if i < len(axes):
        ax = axes[i]
        data_to_plot = [customer_df[customer_df['Cluster'] == c][feature].values for c in range(best_k)]
        bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, best_k))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(feature)
        ax.set_xlabel('Cluster')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([f'C{i}' for i in range(best_k)])

for i in range(len(features), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Feature Distributions Across Clusters', fontsize=16)
plt.tight_layout()
plt.savefig('3_feature_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: 3_feature_distributions.png")
plt.close()

# Figure 4: Silhouette Plot
fig4, ax = plt.subplots(figsize=(12, 8))

silhouette_vals = silhouette_samples(X_pca, clusters)
y_lower = 10

for i in range(best_k):
    cluster_silhouette = silhouette_vals[clusters == i]
    cluster_silhouette.sort()
    size = len(cluster_silhouette)
    y_upper = y_lower + size
    
    color = plt.cm.viridis(i / best_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + size/2, f'Cluster {i}', size=12, ha='right', va='center')
    y_lower = y_upper + 10

ax.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--', 
           linewidth=2, label=f'Average Silhouette: {np.mean(silhouette_vals):.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot - Cluster Cohesion & Separation')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 1)

plt.tight_layout()
plt.savefig('4_silhouette_plot.png', dpi=300, bbox_inches='tight')
print("Saved: 4_silhouette_plot.png")
plt.close()

# Figure 5: Radar Chart
fig5, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

normalized = scaler.transform(centroids_scaled)
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

for i in range(best_k):
    values = normalized[i].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, size=10)
ax.set_title('Persona Radar Chart', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig('5_radar_chart.png', dpi=300, bbox_inches='tight')
print("Saved: 5_radar_chart.png")
plt.close()

# Figure 6: Summary Table
fig6, ax = plt.subplots(figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

summary_data = []
for i in range(best_k):
    centroid_features = centroid_df.iloc[i][features].values
    row = [f'Cluster {i}', cluster_counts[i], f'{cluster_counts[i]/len(customer_df)*100:.1f}%']
    diff = (centroid_features / overall_avg - 1) * 100
    top_indices = np.argsort(np.abs(diff))[::-1][:3]
    for idx in top_indices:
        row.append(f'{features[idx]}: {centroid_features[idx]:.1f}')
    summary_data.append(row)

cols = ['Cluster', 'Size', '% of Total']
for i in range(3):
    cols.append(f'Key Feature {i+1}')

table = ax.table(cellText=summary_data, colLabels=cols,
                cellLoc='center', loc='center',
                colWidths=[0.08, 0.08, 0.08, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.title('Customer Segment Summary', fontsize=14)
plt.tight_layout()
plt.savefig('6_cluster_summary.png', dpi=300, bbox_inches='tight')
print("Saved: 6_cluster_summary.png")
plt.close()

# ----------------------------------------------------------------------
# 12. SAVE OUTPUTS
# ----------------------------------------------------------------------
print("\nSTEP 12: SAVING OUTPUTS")
print("-" * 40)

customer_df['Persona'] = customer_df['Cluster'].map(
    {i: persona_descriptions[i][1] for i in range(best_k)}
)

customer_df.to_csv('customer_segments_with_personas.csv', index=False)
print("Saved: customer_segments_with_personas.csv")

centroid_df.to_csv('cluster_centroids_summary.csv')
print("Saved: cluster_centroids_summary.csv")

feature_stats = customer_df[features + ['Cluster']].groupby('Cluster').agg(['mean', 'std', 'count']).round(2)
feature_stats.to_csv('cluster_feature_stats.csv')
print("Saved: cluster_feature_stats.csv")

# ----------------------------------------------------------------------
# 13. FINAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("PROJECT 3 COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nOUTPUT FILES GENERATED:")
print("-" * 60)
print("Visualizations:")
print("  1. 1_elbow_silhouette.png       - Elbow & Silhouette plots")
print("  2. 2_clusters_2d.png            - 2D cluster visualization")
print("  3. 3_feature_distributions.png  - Boxplots by cluster")
print("  4. 4_silhouette_plot.png        - Silhouette coefficient plot")
print("  5. 5_radar_chart.png            - Persona radar chart")
print("  6. 6_cluster_summary.png        - Cluster summary table")
print("\nData Files:")
print("  7. customer_segments_with_personas.csv  - Full customer data with clusters & personas")
print("  8. cluster_centroids_summary.csv        - Centroids in original units")
print("  9. cluster_feature_stats.csv            - Feature statistics by cluster")

print("\n" + "=" * 80)
print("BUSINESS PERSONAS SUMMARY")
print("=" * 80)

for i, (idx, persona, desc) in enumerate(persona_descriptions):
    size = cluster_counts[idx]
    pct = size/len(customer_df)*100
    print(f"\n{persona}")
    print(f"  Size: {size:,} customers ({pct:.1f}%)")
    print(f"  {desc}")
    print("  Key traits:")
    centroid_features = centroid_df.iloc[idx][features].values
    diff = (centroid_features / overall_avg - 1) * 100
    top_indices = np.argsort(np.abs(diff))[::-1][:3]
    for j in top_indices:
        print(f"    - {features[j]}: {centroid_features[j]:.2f} ({abs(diff[j]):.1f}% {'above' if diff[j] > 0 else 'below'} avg)")

print("\n" + "=" * 80)
print("RECOMMENDED ACTIONS")
print("=" * 80)

for i, (idx, persona, desc) in enumerate(persona_descriptions):
    print(f"\n{persona}:")
    if "VIP" in persona or "Big Spenders" in persona:
        print("  - Implement VIP loyalty program")
        print("  - Offer exclusive early access")
        print("  - Send personalized thank-you gifts")
    elif "Churn" in persona:
        print("  - Send re-engagement email campaigns")
        print("  - Offer special win-back discounts")
        print("  - Survey to understand reasons for decreased engagement")
    elif "Deal" in persona:
        print("  - Target with discount promotions")
        print("  - Send flash sale alerts")
        print("  - Bundle products to increase order value")
    elif "Regular" in persona:
        print("  - Maintain high service quality")
        print("  - Encourage reviews and referrals")
        print("  - Cross-sell complementary products")
    elif "Explorer" in persona:
        print("  - Showcase new product arrivals")
        print("  - Create curated collections")
        print("  - Implement recommendation engine")
    elif "Average" in persona:
        print("  - Focus on moving to higher-value segments")
        print("  - Test promotional strategies")
        print("  - Analyze conversion barriers")

print("\n" + "=" * 80)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 80)