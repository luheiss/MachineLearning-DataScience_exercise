import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# ---  Daten laden ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Für die Übung nur 500 Bilder verwenden
n_samples = 500
X = mnist.data[:n_samples]
y = mnist.target[:n_samples].astype(int)

# --- Normalisieren auf [0,1] ---
X = X.astype(np.float32) / 255.0

# --- Rauschen hinzufügen ---
np.random.seed(42)  # für reproduzierbare Ergebnisse
noise_factor = 0.3
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
X_noisy = np.clip(X_noisy, 0., 1.)  # Werte zwischen 0 und 1 halten

# --- Visualisieren Original vs. Noisy ---
n_display = 5
plt.figure(figsize=(10, 4))
for i in range(n_display):
    # Original
    plt.subplot(2, n_display, i+1)
    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Original")
    
    # Noisy
    plt.subplot(2, n_display, n_display + i + 1)
    plt.imshow(X_noisy[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Noisy")
plt.show()

# ---  PCA auf den flachgemachten Bildern trainieren ---
X_noisy_flat = X_noisy.reshape(n_samples, -1)  # flatten 28x28 -> 784
pca = PCA(n_components=100)  # Anzahl der Komponenten
X_pca = pca.fit_transform(X_noisy_flat)

# --- PCA inverse transform für Denoising ---
X_denoised_flat = pca.inverse_transform(X_pca)
X_denoised = X_denoised_flat.reshape(n_samples, 28, 28)

# --- Visualisierung Original vs. Noisy vs. Denoised ---
plt.figure(figsize=(15, 6))
for i in range(n_display):
    # Original
    plt.subplot(3, n_display, i+1)
    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Original")
    
    # Noisy
    plt.subplot(3, n_display, n_display + i + 1)
    plt.imshow(X_noisy[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Noisy")
    
    # Denoised
    plt.subplot(3, n_display, 2*n_display + i + 1)
    plt.imshow(X_denoised[i], cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Denoised")
plt.show()
