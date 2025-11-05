import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from palmerpenguins import load_penguins

# --- Daten laden und vorbereiten ---
peng = load_penguins()
df_cleaned = peng.dropna()

# Wir nehmen nur zwei Features für Regression: bill_length_mm -> body_mass_g
X = df_cleaned[['bill_length_mm']].values
y = df_cleaned['body_mass_g'].values

# Trainings- und Testset splitten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- (a) Gradient Boosting manuell auf Residuen ---
# Level 1: trainiere ersten Decision Tree auf Originaldaten
tree1 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree1.fit(X_train, y_train)
y_pred1 = tree1.predict(X_train)

# Residuen berechnen
residuals1 = y_train - y_pred1

# Level 2: trainiere auf Residuen
tree2 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree2.fit(X_train, residuals1)
y_pred2 = tree2.predict(X_train)

# Level 3
residuals2 = residuals1 - y_pred2
tree3 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree3.fit(X_train, residuals2)
y_pred3 = tree3.predict(X_train)

# Level 4
residuals3 = residuals2 - y_pred3
tree4 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree4.fit(X_train, residuals3)
y_pred4 = tree4.predict(X_train)

# Level 5
residuals4 = residuals3 - y_pred4
tree5 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree5.fit(X_train, residuals4)
y_pred5 = tree5.predict(X_train)

# Gesamtvorhersage auf Trainingsdaten
y_pred_train = y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5

mse_train = mean_squared_error(y_train, y_pred_train)
print(f"MSE Training (manuell): {mse_train:.2f}")

# --- (b) Gradient Boosting mit sklearn ---
gbr = GradientBoostingRegressor(
    n_estimators=5,  # 5 Level
    max_depth=3,
    random_state=42
)
gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_gbr)
print(f"MSE Test (GradientBoostingRegressor): {mse_test:.2f}")

# --- Visualisierung ---
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.plot(np.sort(X_train[:,0]), gbr.predict(np.sort(X_train[:,0]).reshape(-1,1)), color='green', label='Gradient Boosting')
plt.xlabel("bill_length_mm")
plt.ylabel("body_mass_g")
plt.title("Gradient Boosting Regression auf Palmer Penguins")
plt.legend()
plt.show()
