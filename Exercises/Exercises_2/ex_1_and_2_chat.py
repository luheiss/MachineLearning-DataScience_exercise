import palmerpenguins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Hinzugefügt für numerische Operationen

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#needed Chatgpt for adding the street plot to the SVC plot


# --- Datenvorbereitung (aus Ihrem Code) ---
peng = palmerpenguins.load_penguins()
df = pd.DataFrame(peng)
cleaned_df = df.dropna()

df_cleaned = cleaned_df[['species', 'bill_depth_mm', 'flipper_length_mm']]
# Reduktion auf zwei Spezies: Adelie und Gentoo
df_fin = df_cleaned[df_cleaned['species'] != 'Chinstrap'].copy() 

# Merkmale und Zielvariable
features = df_fin[['bill_depth_mm', 'flipper_length_mm']]
X = features.values
# 0 für Adelie, 1 für Gentoo (Da 'Gentoo' != 'Chinstrap' hier die 2. Spezies ist)
y = (df_fin['species'] == 'Gentoo').astype(int).values 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42,    
    stratify=y          
)

# Soft Margin Klassifikation ohne StandardScaler
model = LinearSVC(C=1.0, random_state=42, dual='auto') # dual='auto' für neuere sklearn Versionen
model.fit(X_train, y_train)

# --- Plot-Erweiterung: Entscheidungsgrenze und Street ---

# Die Entscheidungsgrenze wird durch die Gleichung w_0 * x_0 + w_1 * x_1 + b = 0 definiert,
# wobei x_0 die bill_depth_mm und x_1 die flipper_length_mm ist.
# Die Street-Grenzen sind durch w_0 * x_0 + w_1 * x_1 + b = 1 und = -1 definiert.

# Koeffizienten und Intercept extrahieren
w = model.coef_[0]
b = model.intercept_[0]

# Definiere die X-Achsen-Werte (bill_depth_mm) für die Darstellung der Geraden
# Wir verwenden den gesamten Datenbereich, um sicherzustellen, dass die Linie lang genug ist
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x0 = np.linspace(x_min, x_max, 100)

# Funktion zur Berechnung der Y-Werte (flipper_length_mm) für eine gegebene 'k'
# x_1 = (-w_0 * x_0 - b + k) / w_1
# Für die Entscheidungsgrenze ist k = 0
# Für die Ränder ist k = 1 (obere Linie) und k = -1 (untere Linie)

# Entscheidungsgrenze (k=0): w[0]*x0 + w[1]*x1 + b = 0
decision_boundary = (-w[0] * x0 - b) / w[1]

# Obere Margin-Grenze (k=1): w[0]*x0 + w[1]*x1 + b = 1
margin_up = (-w[0] * x0 - b + 1) / w[1]

# Untere Margin-Grenze (k=-1): w[0] * x0 + w[1] * x1 + b = -1
margin_down = (-w[0] * x0 - b - 1) / w[1]

# --- Plot erstellen ---

plt.figure(figsize=(10, 6))

# Datenpunkte plotten
# Verwenden Sie das Original-DataFrame für die Spezies-Farben
sns.scatterplot(
    data=df_fin,
    x='bill_depth_mm',
    y='flipper_length_mm',
    hue='species',
    style='species',
    palette={'Adelie': 'blue', 'Gentoo': 'green'},
    s=70,
    zorder=2 # Sicherstellen, dass die Punkte über dem Street-Bereich liegen
)

# Entscheidungsgrenze plotten (durchgezogene Linie, k=0)
plt.plot(x0, decision_boundary, 'k-', linewidth=2, label='Entscheidungsgrenze (Soft Margin)')

# Street-Grenzen plotten (gestrichelte Linien, k=1 und k=-1)
plt.plot(x0, margin_up, 'k--', linewidth=1, label='Margin Grenze')
plt.plot(x0, margin_down, 'k--', linewidth=1)

# Schattierte Fläche für die "Street" (Marge) zwischen den Rändern
plt.fill_between(x0, margin_down, margin_up, color='gray', alpha=0.2, label='Street (Marge)')

# Achsenbeschriftungen und Titel
plt.xlabel('Schnabeltiefe (bill_depth_mm)')
plt.ylabel('Flossenlänge (flipper_length_mm)')
plt.title('LinearSVC Soft Margin Klassifikation (Adelie vs. Gentoo) ohne Skalierung')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- LinearSVC (ohne StandardScaler) Ergebnisse ---")
predictions = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions)) 
print("\nClassification Report:")
print(classification_report(y_test, predictions))