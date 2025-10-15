import palmerpenguins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

peng= palmerpenguins.load_penguins()
df = pd.DataFrame(peng)


cleaned_df = df.dropna()

sns.pairplot(cleaned_df, hue='species')
# Zeigt den Plot an
#plt.show()

# notizen: 
# - Zum trenne der Arten ist es wahrscheinlich am besten flipper_length_mm und bill_length_mm zu verwenden
# oder bill_depth_mm und bill_length_mm

## Task 2:
df_cleaned = cleaned_df[['species', 'bill_depth_mm', 'flipper_length_mm']]
df_fin = df_cleaned[df_cleaned['species'] != 'Chinstrap'].copy()


sns.pairplot(df_fin, hue='species')


# notizen: 
# yes we can seperate them with a hard classifier
# a possible classifier could be: if flipper_length_mm > 200 then Gentoo else Adelie

from sklearn.svm import LinearSVC # Für die Soft Margin Klassifikation
from sklearn.preprocessing import StandardScaler # Für die Datenstandardisierung
from sklearn.model_selection import train_test_split # Für das Aufteilen in Trainings- und Testdaten
from sklearn.metrics import classification_report, confusion_matrix # Für die Auswertung

sklearn_df = df_fin.copy()

features = sklearn_df[['bill_depth_mm', 'flipper_length_mm']]
# Korrektur: X ist einfach features (umgewandelt in NumPy-Array)
X = features.values
labels = sklearn_df['species']

y = (df_fin['species'] == 'Gentoo').astype(int).values 
# oder mit der korrekten Variablen aus der Datei:
# y = (df_fin[TARGET] == 'Gentoo').astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20 % for the test
    random_state=42,     # Feste Zufallszahl für reproduzierbare Ergebnisse
    stratify=y           # Wichtig: Stellt sicher, dass das Verhältnis von Adelie/Gentoo in beiden Sets gleich ist
)

model = LinearSVC(C=1.0)  # C ist der Regularisierungsparameter
model.fit(X_train, y_train)  # X_train: Features, y_train: Zielwerte
predictions = model.predict(X_test)  # X_test: Testdaten




# mit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearSVC(C=1.0)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)



if __name__ == "__main__":
    plt.show()
    
    print(df)
    
    print(cleaned_df)
    print(cleaned_df.info())
    print(df_fin)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))   
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix (mit Scaler):")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report (mit Scaler):")
    print(classification_report(y_test, predictions))