from ex_1_and_2 import df_fin
from ex_1_and_2 import X, y, X_train, X_test, y_train, y_test
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


print(df_fin)

# Degree 3 als Startwert für die Visualisierung in Teil (b)
poly_svc_pipeline = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearSVC(C=1.0, random_state=42, dual='auto'))

poly_svc_pipeline.fit(X_train, y_train)

score = poly_svc_pipeline.score(X_test, y_test)
print(f"Genauigkeit der Polynomial SVC (Degree 3): {score:.4f}")