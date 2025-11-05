from sklearn.metrics import accuracy_score
import sklearn.ensemble as en
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


peng = load_penguins()
df_cleaned = peng.dropna()
features = df_cleaned[['bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_length_mm']].values

X = features
y = df_cleaned['species'].map({'Adelie':0, 'Gentoo':1, 'Chinstrap':2}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, 
    random_state=42,
    stratify=y
)

model_SVC = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, random_state=42, probability=True)
)
model_DT = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier(max_depth=5, random_state=42)
)
model_LR = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model_SVC.fit(X_train, y_train)
model_DT.fit(X_train, y_train)
model_LR.fit(X_train, y_train)

print("Einzelmodell Genauigkeiten:")
print("SVC:", accuracy_score(y_test, model_SVC.predict(X_test)))
print("Decision Tree:", accuracy_score(y_test, model_DT.predict(X_test)))
print("Logistic Regression:", accuracy_score(y_test, model_LR.predict(X_test)))

estimators = [
    ('svc', model_SVC),
    ('dt', model_DT),
    ('lr', model_LR)
]
#print("estimators:", estimators)

# Soft Voting
voting_clf_soft = en.VotingClassifier(
    estimators=estimators, 
    voting='soft'
)
voting_clf_soft.fit(X_train, y_train)
y_pred_soft = voting_clf_soft.predict(X_test)
score_soft = accuracy_score(y_test, y_pred_soft)
print(f"Genauigkeit VotingClassifier (Soft): {score_soft:.4f}")

# Hard Voting
voting_clf_hard = en.VotingClassifier(
    estimators=estimators, 
    voting='hard'
)
voting_clf_hard.fit(X_train, y_train)
y_pred_hard = voting_clf_hard.predict(X_test)
score_hard = accuracy_score(y_test, y_pred_hard)
print(f"Genauigkeit VotingClassifier (Hard): {score_hard:.4f}")
