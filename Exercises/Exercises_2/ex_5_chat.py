import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import math

# --- (5) Linear Regression ---

# Gegebene Datenpunkte
# X ist der unabhängige Vektor (x-Achse)
x = np.arange(1, 11).reshape(-1, 1)
# Y und Z sind die abhängigen Vektoren (y-Achse)
y = np.array([0.2, 0.5, 0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)
z = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)

# h(x) = Theta[1] + Theta[0] * x
# np.polyval erwartet die Koeffizienten in der Reihenfolge [Theta_m, ..., Theta_1, Theta_0]
# Daher verwenden wir Theta[0] für die Steigung und Theta[1] für den Achsenabschnitt.

# --- (a) Erstellung der Norm-Funktionen ---

# E1(h) = 1/m * Summe |h(x) - y| (Mean Absolute Error, MAE)
def fit_E1(Theta, x, y):
    # h(x) = Theta[1] + Theta[0] * x
    predictions = Theta[1] + Theta[0] * x
    error = np.mean(np.abs(predictions - y))
    return error

# E2(h) = 1/m * Summe |h(x) - y|^2 (Least Square Error, MSE)
def fit_E2(Theta, x, y):
    predictions = Theta[1] + Theta[0] * x
    error = np.mean((predictions - y)**2)
    return error

# E_inf(h) = max |h(x) - y| (Maximum Norm Error)
def fit_E_inf(Theta, x, y):
    predictions = Theta[1] + Theta[0] * x
    error = np.max(np.abs(predictions - y))
    return error

# --- (b) Initial Guess (Startwert) ---

# Theta = [Steigung, Achsenabschnitt] = [Theta0, Theta1] in der Aufgabenstellung
# h(xi) = Θ1 + Θ0xi
# Da polyval [Steigung, Achsenabschnitt] erwartet:
Theta0_initial = np.array([1.0, 1.0]) 

# --- (c) & (d) & (e) Optimierung und Visualisierung für Daten Y ---

# Optimierung für Y
p1_y = fmin(fit_E1, Theta0_initial, args=(x, y), disp=0) # E1-Norm (L1)
p2_y = fmin(fit_E2, Theta0_initial, args=(x, y), disp=0) # E2-Norm (L2)
pInf_y = fmin(fit_E_inf, Theta0_initial, args=(x, y), disp=0) # E_inf-Norm (L_inf)

# Erstellen von Stützstellen für die Plot-Geraden
x_plot = np.arange(0, 11, 0.1)

plt.figure(figsize=(12, 6))

# Datenpunkte plotten
plt.plot(x, y, 'ko', label='Datenpunkte Y')

# Regressionsgeraden plotten
# np.polyval(p, x) wertet das Polynom mit Koeffizienten p an den Stellen x aus.
# p wird hier als [Steigung, Achsenabschnitt] übergeben.
plt.plot(x_plot, np.polyval(p1_y, x_plot), 'r-', label=f'$E_1$ (MAE): $\\Theta_0={p1_y[0]:.2f}, \\Theta_1={p1_y[1]:.2f}$')
plt.plot(x_plot, np.polyval(p2_y, x_plot), 'g-', label=f'$E_2$ (MSE): $\\Theta_0={p2_y[0]:.2f}, \\Theta_1={p2_y[1]:.2f}$')
plt.plot(x_plot, np.polyval(pInf_y, x_plot), 'b-', label=f'$E_{'\\infty'}$ (Max): $\\Theta_0={pInf_y[0]:.2f}, \\Theta_1={pInf_y[1]:.2f}$')

plt.title('Lineare Regression - Vergleich der Normen auf Daten Y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --- (f) Redo mit Daten Z (mit Outlier) ---

# Optimierung für Z
p1_z = fmin(fit_E1, Theta0_initial, args=(x, z), disp=0) # E1-Norm (L1)
p2_z = fmin(fit_E2, Theta0_initial, args=(x, z), disp=0) # E2-Norm (L2)
pInf_z = fmin(fit_E_inf, Theta0_initial, args=(x, z), disp=0) # E_inf-Norm (L_inf)

plt.figure(figsize=(12, 6))

# Datenpunkte plotten
plt.plot(x, z, 'mo', label='Datenpunkte Z (mit Outlier bei x=4)')

# Regressionsgeraden plotten
plt.plot(x_plot, np.polyval(p1_z, x_plot), 'r-', label=f'$E_1$ (MAE): $\\Theta_0={p1_z[0]:.2f}, \\Theta_1={p1_z[1]:.2f}$')
plt.plot(x_plot, np.polyval(p2_z, x_plot), 'g-', label=f'$E_2$ (MSE): $\\Theta_0={p2_z[0]:.2f}, \\Theta_1={p2_z[1]:.2f}$')
plt.plot(x_plot, np.polyval(pInf_z, x_plot), 'b-', label=f'$E_{'\\infty'}$ (Max): $\\Theta_0={pInf_z[0]:.2f}, \\Theta_1={pInf_z[1]:.2f}$')

plt.title('Lineare Regression - Vergleich der Normen auf Daten Z (mit Outlier)')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.grid(True)
plt.show()