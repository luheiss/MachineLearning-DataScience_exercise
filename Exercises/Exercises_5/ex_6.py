import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Load the dataset (requires palmerpenguins library or downloading the csv)
# Using seaborn's built-in version for convenience
penguins = sns.load_dataset("penguins").dropna()
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
true_labels = penguins['species']

# Standardize the data
# Clustering algorithms are sensitive to scales (especially body mass in grams vs lengths in mm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering on original scaled data
# We know there are 3 species, so we set n_clusters=3
kmeans_orig = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_orig = kmeans_orig.fit_predict(X_scaled)

# DBSCAN clustering on original scaled data
# eps and min_samples need tuning; these are starting values
dbscan_orig = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan_orig = dbscan_orig.fit_predict(X_scaled)

# Apply PCA
# Reducing to 2 components for visualization and noise reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Means clustering on PCA-transformed data
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_pca = kmeans_pca.fit_predict(X_pca)

# DBSCAN clustering on PCA-transformed data
dbscan_pca = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan_pca = dbscan_pca.fit_predict(X_pca)

# Visualization and Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot K-Means Original
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans_orig, cmap='viridis')
axes[0, 0].set_title("K-Means (Original Scaled Data)")

# Plot DBSCAN Original
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan_orig, cmap='plasma')
axes[0, 1].set_title("DBSCAN (Original Scaled Data)")

# Plot K-Means PCA
axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans_pca, cmap='viridis')
axes[1, 0].set_title("K-Means (On PCA Data)")

# Plot DBSCAN PCA
axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan_pca, cmap='plasma')
axes[1, 1].set_title("DBSCAN (On PCA Data)")

plt.tight_layout()
plt.show()

# Evaluation using Adjusted Rand Index (ARI)
# 1.0 is a perfect match with true species, 0.0 is random
print(f"K-Means Original ARI: {adjusted_rand_score(true_labels, labels_kmeans_orig):.3f}")
print(f"K-Means PCA ARI:      {adjusted_rand_score(true_labels, labels_kmeans_pca):.3f}")
print(f"DBSCAN Original ARI:  {adjusted_rand_score(true_labels, labels_dbscan_orig):.3f}")
print(f"DBSCAN PCA ARI:       {adjusted_rand_score(true_labels, labels_dbscan_pca):.3f}")