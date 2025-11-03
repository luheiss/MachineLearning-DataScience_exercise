import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Daten
x = np.arange(1, 11).reshape(-1, 1)
y = np.array([0.2, 0.5, 0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)
z = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape(-1, 1)
print(f"Datenpunkt x {x}")

# lineares Modell in x-y-Ebene
pipelr = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
pipelr.fit(x, y)
score_lr = pipelr.score(x, y)
print(f"Genauigkeit Lineares Modell: {score_lr:.4f}")
y_pred_lr = pipelr.predict(x)

# lineares Modell mit Ausreißer in z-Richtung
pipelr_z = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
pipelr_z.fit(x, z)
score_lr_z = pipelr_z.score(x, z)
print(f"Genauigkeit Lineares Modell (Z): {score_lr_z:.4f}")
y_pred_lr_z = pipelr_z.predict(x)

# Visualisierung
plt.figure(figsize=(12, 5)) 
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Datenpunkte')
plt.plot(x, y_pred_lr, color='red', label='Lineares Modell')
plt.title('Lineares Modell in x-y-Ebene')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(x, z, color='blue', label='Datenpunkte mit Ausreißer')
plt.plot(x, y_pred_lr_z, color='red', label='Lineares Modell')
plt.title('Lineares Modell mit Ausreißer in z-Richtung')
plt.xlabel('x') 
plt.ylabel('z')
plt.tight_layout()
plt.show()


# HINWEIS: Speichere die ursprünglichen Daten, bevor x für Moore-Penrose überschrieben wird.
x_orig = x.copy()
y_orig = y.copy()
z_orig = z.copy()

# Using the Moore-Penrose pseudo-inverse (ATA)−1AT that can be computed by np.linalg.pinv as described in the lecture with closed-form solution.
x = np.array([x, np.ones(x.shape)]).T
b_y = np.linalg.pinv(x).dot(y)  #in this case b_y contains the slope and the intercept of the linear regression line
b_z = np.linalg.pinv(x).dot(z)  #in this case b_z contains the slope and the intercept of the linear regression line
print(f"Moore-Penrose pseudo-inverse coefficients for y: {b_y.ravel()}")
print(f"Moore-Penrose pseudo-inverse coefficients for z: {b_z.ravel()}")

# visualization
y_pred_lr_pinv = x.dot(b_y)
z_pred_lr_pinv = x.dot(b_z)

# Plot 1: Lineares Modell (Moore-Penrose)
plt.subplot(1, 2, 1)
# KORREKTUR: Verwende x_orig für X-Werte und .flatten() auf allen Arrays.
plt.scatter(x_orig.flatten(), y_orig.flatten(), color='blue', label='Datenpunkte')
plt.plot(x_orig.flatten(), y_pred_lr_pinv.flatten(), color='green', label='Lineares Modell (Moore-Penrose)')
plt.title('Lineares Modell in x-y-Ebene (Moore-Penrose)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() # Legende hinzugefügt

# Plot 2: Lineares Modell mit Ausreißer (Moore-Penrose)
plt.subplot(1, 2, 2)
# KORREKTUR: Verwende x_orig für X-Werte und .flatten() auf allen Arrays.
plt.scatter(x_orig.flatten(), z_orig.flatten(), color='blue', label='Datenpunkte mit Ausreißer')
plt.plot(x_orig.flatten(), z_pred_lr_pinv.flatten(), color='green', label='Lineares Modell (Moore-Penrose)')
plt.title('Lineares Modell mit Ausreißer in z-Richtung (Moore-Penrose)')
plt.xlabel('x')
plt.ylabel('z')
plt.legend() # Legende hinzugefügt

plt.tight_layout()
plt.show()

# Not all models are linear, is there a way to handle the following hypothesis function? h(xi,Θ) = Θ2exp(Θ1xi) = yi
def model_func(x, theta1, theta2):
    return theta2 * np.exp(theta1 * x)

# Fit the model to the data using curve_fit
initial_guess = [0.1, 0.1]
# KORREKTUR 1: Verwende x_orig.ravel() anstelle von x.ravel()
params, covariance = optimize.curve_fit(model_func, x_orig.ravel(), y_orig.ravel(), p0=initial_guess)
print(f"Fitted parameters: theta1 = {params[0]}, theta2 = {params[1]}")
# KORREKTUR 2: Verwende x_orig.ravel() anstelle von x.ravel() für die Vorhersage
y_pred_nonlinear = model_func(x_orig.ravel(), *params)

# Visualisierung
plt.figure(figsize=(6, 5))
# KORREKTUR 3: Verwende x_orig und y_orig für den Scatter-Plot
plt.scatter(x_orig.flatten(), y_orig.flatten(), color='blue', label='Datenpunkte')
# KORREKTUR 4: Verwende x_orig.flatten() für den Plot der Linie
plt.plot(x_orig.flatten(), y_pred_nonlinear, color='orange', label='Nicht-lineares Modell')
plt.title('Nicht-lineares Modell in x-y-Ebene')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()