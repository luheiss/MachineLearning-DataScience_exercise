import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Daten Setup
X = np.random.rand(100, 2) - 0.5
y = (X[:, 0] > 0).astype(np.int32)

# Rotationsmatrix (45 Grad)
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
X_rot = X @ rotation_matrix

# Vergleich: Original vs. Rotiert mit verschiedenen Random States
def train_and_plot(X_data, title):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_data, y)
    plt.figure(figsize=(6,4))
    plot_tree(clf, filled=True, feature_names=["x1", "x2"])
    plt.title(title)
    plt.show()

train_and_plot(X, "Tree: Original")
train_and_plot(X_rot, "Tree: Rotated")

# Lösung: Pipeline mit PCA
# PCA findet die Hauptkomponenten (neue Achsen), die oft die Rotation korrigieren
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    DecisionTreeClassifier(random_state=42)
)

pipeline.fit(X_rot, y)

# Visualisierung des Pipeline-Trees
plt.figure(figsize=(6,4))
plot_tree(pipeline.named_steps['decisiontreeclassifier'], filled=True)
plt.title("Tree after PCA Transformation (Rotated)")
plt.show()