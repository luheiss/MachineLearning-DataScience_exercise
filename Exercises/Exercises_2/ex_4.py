from ex_1_and_2 import df_fin
from ex_1_and_2 import X, y, X_train, X_test, y_train, y_test
# SVC importieren für Kernel-Methoden
from sklearn.svm import LinearSVC, SVC 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

print(df_fin)

# --- Modell 1: Explizite Polynom-Features (Ihr Original) ---
poly_feat_pipeline = make_pipeline(
    StandardScaler(), # Scaling sollte VOR PolynomialFeatures stehen
    PolynomialFeatures(degree=3), 
    LinearSVC(C=1.0, random_state=42, dual='auto')
)
poly_feat_pipeline.fit(X_train, y_train)
score_feat = poly_feat_pipeline.score(X_test, y_test)
print(f"Genauigkeit Explizite Features (Degree 3): {score_feat:.4f}")

# --- NEU: Modell 2: Polynomial Kernel SVC (Kernel Trick) ---
# Kernel Trick: Verwendet SVC mit kernel='poly'. 
# Die Transformation wird implizit durchgeführt.
poly_kernel_pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='poly', degree=3, C=1.0, random_state=42)            # or 'rbf' for RBF kernel instead of 'poly'
)
poly_kernel_pipeline.fit(X_train, y_train)
score_kernel = poly_kernel_pipeline.score(X_test, y_test)
print(f"Genauigkeit Polynomial Kernel (Degree 3): {score_kernel:.4f}")


# --- Visualisierung der beiden Modelle ---

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

plt.figure(figsize=(18, 7)) # Größere Figur für 2 Subplots

# --- PLOT 1: Explizite Polynom-Features ---
ax1 = plt.subplot(1, 2, 1)

DecisionBoundaryDisplay.from_estimator(
    poly_feat_pipeline,
    X_all, 
    ax=ax1,
    plot_method="contourf", # ist standardmäßig "contourf"
    cmap=plt.cm.coolwarm, # Farbkarte für die Konturen
    alpha=0.6, # Transparenz der Konturen
    response_method="predict" # oder "decision_function" oder "predict_proba", predict ist für Klassifikation passend
)

ax1.scatter(
    X_all[:, 0], 
    X_all[:, 1], 
    c=y_all, 
    cmap=plt.cm.coolwarm, 
    s=60, 
    edgecolors='k',
)

ax1.set_xlabel('Bill Length (mm)')
ax1.set_ylabel('Flipper Length (mm)')
ax1.set_title(f'Explizite Polynom-Features (Degree 3, Score: {score_feat:.4f})')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.5)


# --- PLOT 2: Polynomial Kernel SVC (Kernel Trick) ---
ax2 = plt.subplot(1, 2, 2)

DecisionBoundaryDisplay.from_estimator(
    poly_kernel_pipeline,
    X_all, 
    ax=ax2,
    plot_method="contourf",
    cmap=plt.cm.coolwarm,
    alpha=0.6,
    response_method="predict"
)

ax2.scatter(
    X_all[:, 0], 
    X_all[:, 1], 
    c=y_all, 
    cmap=plt.cm.coolwarm, 
    s=60, 
    edgecolors='k',
)

ax2.set_xlabel('Bill Length (mm)')
ax2.set_ylabel('Flipper Length (mm)')
ax2.set_title(f'Polynomial Kernel (Degree 3) [Kernel Trick], Score: {score_kernel:.4f}')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()