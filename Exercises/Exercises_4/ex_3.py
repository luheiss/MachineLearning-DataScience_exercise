from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt


# Daten laden und vorbereiten
peng = load_penguins()
df_cleaned = peng.dropna()

# Nur zwei Klassen auswählen (z.B. Adelie und Gentoo)
df_binary = df_cleaned[df_cleaned['species'].isin(['Adelie', 'Gentoo'])]

X = df_binary[['bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_length_mm']].values
y = df_binary['species'].map({'Adelie': 0, 'Gentoo': 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Basismodelle definieren
base_SVC = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, random_state=42, probability=True)
)
base_DT = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier(max_depth=5, random_state=42)
)
base_LR = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Bagging Classifier für jedes Modell
bag_SVC = BaggingClassifier(
    estimator=base_SVC,
    n_estimators=10,
    random_state=42
)
bag_DT = BaggingClassifier(
    estimator=base_DT,
    n_estimators=10,
    random_state=42
)
bag_LR = BaggingClassifier(
    estimator=base_LR,
    n_estimators=10,
    random_state=42
)

# Modelle trainieren
bag_SVC.fit(X_train, y_train)
bag_DT.fit(X_train, y_train)
bag_LR.fit(X_train, y_train)

# Genauigkeiten prüfen
for name, model in [('Bagging SVC', bag_SVC), ('Bagging DT', bag_DT), ('Bagging LR', bag_LR)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Genauigkeit = {acc:.4f}")

# Modelle kombinieren via Voting
voting_bag = VotingClassifier(
    estimators=[
        ('bag_svc', bag_SVC),
        ('bag_dt', bag_DT),
        ('bag_lr', bag_LR)
    ],
    voting='soft'
)
voting_bag.fit(X_train, y_train)
y_pred_voting = voting_bag.predict(X_test)
score_voting = accuracy_score(y_test, y_pred_voting)
print(f"Genauigkeit VotingClassifier (kombinierte Bagging-Modelle): {score_voting:.4f}")

# Visualisierung (nur zwei Features für 2D-Darstellung)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_voting, cmap='coolwarm', edgecolor='k')
plt.title("Vorhersagen des VotingClassifier (Bagging-Modelle)")
plt.xlabel("bill_depth_mm")
plt.ylabel("flipper_length_mm")
plt.show()
