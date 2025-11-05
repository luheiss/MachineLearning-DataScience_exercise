import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

# -- MNIST Daten laden ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Kleine Menge für schnelle Berechnung
n_samples = 1000
X = mnist.data[:n_samples]
y = mnist.target[:n_samples].astype(int)

# Normalisieren auf [0,1] 
X = X.astype(np.float32) / 255.0

# KMeans trainieren
np.random.seed(42)  # Reproduzierbarkeit
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

#Clusterzentren visualisieren
plt.figure(figsize=(10, 4))
for i in range(10):
    center = kmeans.cluster_centers_[i].reshape(28,28)  # reshape auf 28x28
    plt.subplot(2, 5, i+1)
    plt.imshow(center, cmap='gray')
    plt.axis('off')
    plt.title(f'Cluster {i}')
plt.suptitle("K-Means Clusterzentren")
plt.show()

#Kurze Diskussion
print("Die Clusterzentren sind 'durchschnittliche' Bilder der jeweiligen Gruppe.")
print("Sie sehen oft wie verschwommene Zahlen aus, nicht wie einzelne Beispiele.")
print("Unterschied zu Lecture: In der Vorlesung wurden evtl. echte Vertreter (Sample-Bilder) genutzt, hier sind es Mittelwerte.")
