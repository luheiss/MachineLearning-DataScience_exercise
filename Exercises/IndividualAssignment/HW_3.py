from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np


# +++++++++++++++++++++++++++ Q3 - Olivetti Faces Clustering with GMM +++++++++++++++++++++++++++
"""
We load the Olivetti faces dataset, apply PCA for dimensionality reduction,
and then use Gaussian Mixture Models (GMM) for clustering. PCA is used to reduce dimensionality 
and therefore improve the time. We also generate new faces and perform anomaly detection by comparing 
scores of normal and altered faces.

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

"""
PCA (Principal Component Analysis)
----------------------------------
Linear dimensionality reduction using Singular Value Decomposition of the 
data to project it to a lower dimensional space.

Parameters
----------
n_components : float, we keep 95% of variance.
random_state : int = 42, 42 is the seed used by the random number generator.

"""
# --- PCA for dimensionality reduction ---
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

"""
GaussianMixture
-------------------
Gaussian Mixture Model for clustering.
Parameters
----------
n_components : int = 40, Number of mixture components.
random_state : int = 42, 42 is the seed used by the random number generator
"""
GaussianModel = GaussianMixture(n_components=40, random_state=42)

GaussianModel.fit(X_train_pca)
y_predict = GaussianModel.predict(X_test_pca)
accuracy_GMM = accuracy_score(y_test, y_predict)
print(f"GMM accuracy: {accuracy_GMM:}")

# --- Generate new faces ---
n_gen = 10
X_gen_pca, y_gen = GaussianModel.sample(n_samples=n_gen)
X_gen_original = pca.inverse_transform(X_gen_pca)

# gGenerated faces and plot the generated faces
plt.figure(figsize=(12, 3))
for i in range(n_gen):
    plt.subplot(1, n_gen, i + 1)
    plt.imshow(X_gen_original[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.title(f"Gen {i+1}")
plt.suptitle("Generated Faces")
plt.show()

# Density of the faces
densities = GaussianModel.score_samples(X_test_pca)

# Determine threshold for anomaly detection
density_threshold = np.percentile(densities, 10)
# select a face to test
normal_face = X_test[0] 
# Rotate the face 
anomalous_face = np.flipud(normal_face.reshape(64, 64)).reshape(-1)

# Transform both faces using PCA
test_faces_pca = pca.transform([normal_face, anomalous_face])

# Score both faces
test_scores = GaussianModel.score_samples(test_faces_pca)

# Print results
print(f"Threshold: {density_threshold:.2f}")
print(f"Normal face score: {test_scores[0]:.2f} -> Anomaly? {test_scores[0] < density_threshold}")
print(f"Anomalous face score: {test_scores[1]:.2f} -> Anomaly? {test_scores[1] < density_threshold}")

# Visualization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"Normal\nScore: {test_scores[0]:.2f}")
plt.imshow(normal_face.reshape(64, 64), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Anomalous (Flipped)\nScore: {test_scores[1]:.2f}")
plt.imshow(anomalous_face.reshape(64, 64), cmap='gray')
plt.axis('off')

plt.show()