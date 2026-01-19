from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


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



# +++++++++++++++++++++++++++ Q2 - Olivetti Faces using DT +++++++++++++++++++++++++++
"""
Training a Decision Tree Classifier on the Olivetti faces dataset to classify images of different persons.
Select the best cluster size for K-Means using silhouette score analysis.
"""

# --- DT Classifier ---
# max_leaf_nodes = 40 because, 40 diff. peaople
dt_Classifier = DecisionTreeClassifier(random_state=42, max_leaf_nodes= 40)

# --- Train the model ---
dt_Classifier.fit(X_train, y_train)

# --- Predict on test set ---
y_pred_dt = dt_Classifier.predict(X_test)

# --- Calculate accuracy ---
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"DT accuracy: {accuracy_dt:}")

# --- Decision Tree Classifier Pipeline ---
DTKmeans_Pipeline = make_pipeline(
     KMeans(n_clusters=130, random_state=42),
     DecisionTreeClassifier(random_state=42)
)

# --- Train the pipeline ---
DTKmeans_Pipeline.fit(X_train, y_train)

# --- Predict Pipeline ---
y_pred_DTKmeans = DTKmeans_Pipeline.predict(X_test)
accuracy_DTKmeans = accuracy_score(y_test, y_pred_DTKmeans)
print(f"DT Kmeans Pipelime accuracy: {accuracy_DTKmeans:}")

from sklearn.metrics import silhouette_score

# --- Test via silhouette score to find the best cluster size for this DS  ---
# for more Infos: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
k_range = range(20, 161, 10) # Test 20, 30, ..., 160 clusters sizes
silhouette_scores = []

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_train)
    
    # Callculate the score
    score = silhouette_score(X_train, labels)
    silhouette_scores.append(score)
    print(f"The Silhouette Score of k={k}: {score:.4f}")

# --- Plot thee silhouett scores ---
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()