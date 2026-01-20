import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA, PCA

# Daten laden und Rauschen hinzufügen
digits = load_digits()
X = digits.data
y = digits.target

# Künstliches Rauschen hinzufügen
np.random.seed(42)
X_noisy = X + np.random.normal(0, 4, X.shape)

# KernelPCA Modell definieren
# gamma: Steuert die Breite des RBF-Kernels
# alpha: Regularisierungsparameter für die inverse Transformation
kernel_pca = KernelPCA(
    n_components=400,
    kernel="rbf",
    gamma=1e-3,
    fit_inverse_transform=True, # Zwingend erforderlich für Denoising (Rekonstruktion)
    alpha=5e-3,
    random_state=42
)

# Training und Rekonstruktion
X_denoised_kpca = kernel_pca.fit_transform(X_noisy)
X_reconstructed_kpca = kernel_pca.inverse_transform(X_denoised_kpca)

# Vergleich mit Standard-PCA
pca = PCA(n_components=0.95) # Erklärt 95% der Varianz
X_reconstructed_pca = pca.inverse_transform(pca.fit_transform(X_noisy))

# Visualisierung
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(X_noisy[0].reshape(8, 8), cmap="gray")
axes[0].set_title("Noisy Image")
axes[1].imshow(X_reconstructed_pca[0].reshape(8, 8), cmap="gray")
axes[1].set_title("Standard PCA")
axes[2].imshow(X_reconstructed_kpca[0].reshape(8, 8), cmap="gray")
axes[2].set_title("Kernel PCA")
plt.show()