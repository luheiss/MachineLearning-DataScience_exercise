from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# +++++++++++++++++++++++++++ Q1 - Olivetti Faces Clustering with K-Means +++++++++++++++++++++++++++
"""
We load the Olivetti faces dataset, and use K-Means for clustering. We also visualize the faces within a 
selected cluster. Then, we count and print how many images are in each cluster.
"""

# --- loads Olivetti DS ---
data_faces = fetch_olivetti_faces()
X = data_faces.data
y = data_faces.target

# --- Split dataset in training and test set ---
# stratify mixes the classes in both sets equally, so that all persons are represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

#--- Model with k-means clustering ---
# 400 images of 40 different persons = 40 clusters
kmeans = KMeans(n_clusters=40, random_state=42)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_test) 
accuracy_Kmeans = accuracy_score(y_test, y_pred)
print(f"Kmeans accuracy: {accuracy_Kmeans:}")


# --- Function to plot the faces within a cluster ---
var = 0  # Cluster number to visualize
cluster_labels = kmeans.labels_
clusters = X_train[cluster_labels == var]


# --- Counts how many images are in each cluster ---
cluster_counts = np.bincount(kmeans.labels_)
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i} has {count} Images")

# --- Ploting the pictures in cluster var ---
plt.figure(figsize=(10, 2))
for i in range(min(20, len(clusters))):
    plt.subplot(1, 20, i + 1)
    plt.imshow(clusters[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.show()