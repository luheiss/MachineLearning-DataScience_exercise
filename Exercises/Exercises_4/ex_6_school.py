import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

# --- Daten generieren ---
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

X_new = np.linspace(0, 3, 100).reshape(100, 1)

# --- Verschiedene α-Werte testen ---
alphas = [0.01, 0.1, 1]         #one is the maximum we should choose

plt.figure(figsize=(12, 5))

# Ridge Regression
plt.subplot(1, 2, 1)
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    y_pred = ridge.predict(X_new)
    plt.plot(X_new, y_pred, label=f"α={alpha}")
plt.scatter(X, y, color='black', label='Data')
plt.title("Ridge Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# LASSO Regression
plt.subplot(1, 2, 2)
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    y_pred = lasso.predict(X_new)
    plt.plot(X_new, y_pred, label=f"α={alpha}")
plt.scatter(X, y, color='black', label='Data')
plt.title("LASSO Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

# --- Erklärung ---
print("Ridge: L2-Regularisierung, verhindert zu große Koeffizienten.")
print("LASSO: L1-Regularisierung, kann Koeffizienten auf exakt 0 setzen (Feature Selection).")
print("Kleine α -> fast normale lineare Regression, große α -> stärkere Regularisierung.")
