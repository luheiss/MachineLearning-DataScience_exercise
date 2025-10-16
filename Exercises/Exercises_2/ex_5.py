import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Daten
x = np.arange(1, 11).reshape(-1, 1)
y = np.array([0.2, 0.5, 0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)
z = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)

# lineares Modell
def h(theta, x):
    # theta[0] = Steigung, theta[1] = Offset
    return theta[0] * x + theta[1]

# Fehlerfunktionen
def fit1(theta, x, y):  # L1-Norm
    return np.mean(np.abs(h(theta, x) - y))

def fit2(theta, x, y):  # L2-Norm
    return np.mean((h(theta, x) - y)**2)

def fit_inf(theta, x, y):  # L∞-Norm
    return np.max(np.abs(h(theta, x) - y))

# Anfangsschätzung
theta0 = [1, 1]

# Optimierung für y
p1_y = optimize.fmin(fit1, theta0, args=(x, y), disp=False)
p2_y = optimize.fmin(fit2, theta0, args=(x, y), disp=False)
pinf_y = optimize.fmin(fit_inf, theta0, args=(x, y), disp=False)

# Optimierung für z
p1_z = optimize.fmin(fit1, theta0, args=(x, z), disp=False)
p2_z = optimize.fmin(fit2, theta0, args=(x, z), disp=False)
pinf_z = optimize.fmin(fit_inf, theta0, args=(x, z), disp=False)

# Auswertung
x_plot = np.arange(0, 11, 0.1)
h1_y = np.polyval(p1_y[::-1], x_plot)
h2_y = np.polyval(p2_y[::-1], x_plot)
hinf_y = np.polyval(pinf_y[::-1], x_plot)

h1_z = np.polyval(p1_z[::-1], x_plot)
h2_z = np.polyval(p2_z[::-1], x_plot)
hinf_z = np.polyval(pinf_z[::-1], x_plot)

# Visualisierung
fig, ax = plt.subplots(1, 2, figsize=(12, 5))


# --- Plot für y ---
ax[0].scatter(x, y, color='black', label='Daten y')
ax[0].plot(x_plot, h1_y, label='L1-Fit')
ax[0].plot(x_plot, h2_y, label='L2-Fit')
ax[0].plot(x_plot, hinf_y, label='L∞-Fit')
ax[0].set_title('Fits für y')
ax[0].legend()
ax[0].grid(True)

# --- Plot für z ---

ax[1].scatter(x, z, color='black', label='Daten z')
ax[1].plot(x_plot, h1_z, label='L1-Fit')
ax[1].plot(x_plot, h2_z, label='L2-Fit')
ax[1].plot(x_plot, hinf_z, label='L∞-Fit')
ax[1].set_title('Fits für z (mit Ausreißer)')
ax[1].legend()
ax[1].grid(True)

plt.show()

# Ergebnisse ausgeben
print("Parameter für y:")
print(f"L1 : {p1_y}")
print(f"L2 : {p2_y}")
print(f"L∞ : {pinf_y}")

print("\nParameter für z:")
print(f"L1 : {p1_z}")
print(f"L2 : {p2_z}")
print(f"L∞ : {pinf_z}")
