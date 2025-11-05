import numpy as np
import matplotlib.pyplot as plt

# --- Daten ---
x = np.arange(1, 11).reshape(-1, 1)
y = np.array([0.2, 0.5, 0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)
z = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)

# SGD-Funktion (lineare Regression)
def stochastic_gradient_descent(X, y, lr=0.01, epochs=1000, k=1):
    n_samples, n_features = X.shape
    # Gewichte und Bias initialisieren
    w = np.zeros((n_features, 1))
    b = 0
    mse_history = []

    for epoch in range(epochs):
        # zufällige Auswahl von k Stichproben
        idx = np.random.choice(n_samples, k, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]

        # Vorhersage
        y_pred = X_batch @ w + b

        # Fehler
        error = y_batch - y_pred

        # Gradient berechnen
        dw = -2 * X_batch.T @ error / k
        db = -2 * np.sum(error) / k

        # Gewichte aktualisieren
        w = w - lr * dw
        b = b - lr * db

        # Gesamtes MSE auf allen Daten für Verlauf
        mse = np.mean((y - (X @ w + b))**2)
        mse_history.append(mse)

    return w, b, mse_history

# --- (b) SGD mit k=1 und k=5 ---
w1, b1, mse1 = stochastic_gradient_descent(x, y, lr=0.01, epochs=500, k=1)
w5, b5, mse5 = stochastic_gradient_descent(x, y, lr=0.01, epochs=500, k=5)

# --- Plot der Konvergenz ---
plt.figure(figsize=(8,6))
plt.plot(mse1, label='k=1', color='blue')
plt.plot(mse5, label='k=5', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Konvergenz des Stochastic Gradient Descent')
plt.legend()
plt.show()

# --- (c) Vorhersage mit SGD-Modellen ---
y_pred1 = x @ w1 + b1
y_pred5 = x @ w5 + b5

plt.figure(figsize=(8,6))
plt.scatter(x, y, label='Original', color='black')
plt.plot(x, y_pred1, label='SGD k=1', color='blue')
plt.plot(x, y_pred5, label='SGD k=5', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vorhersage mit Stochastic Gradient Descent')
plt.legend()
plt.show()

print(f"Gewichte k=1: w={w1.flatten()}, b={b1:.2f}")
print(f"Gewichte k=5: w={w5.flatten()}, b={b5:.2f}")
