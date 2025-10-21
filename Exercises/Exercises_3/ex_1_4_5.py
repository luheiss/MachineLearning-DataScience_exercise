import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import palmerpenguins
import sklearn.tree as dt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

peng = palmerpenguins.load_penguins()
df= pd.DataFrame(peng)
df_cleaned = df.dropna()
seaborn.pairplot(df_cleaned, hue='species')
#plt.show()

pipe = make_pipeline(
    StandardScaler(),
    dt.DecisionTreeClassifier(max_depth=2, random_state=42)
)

pipe_randomForest = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42,max_depth=3) #3 depth = 0.7910 accuracy
)

features = df_cleaned[['bill_depth_mm', 'flipper_length_mm']]
features_rotated = df_cleaned[['flipper_length_mm', 'bill_depth_mm']]
featuer_all = df_cleaned[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values

X = features.values  #Genauigkeit Decision Tree: 0.7761
#X = features_rotated.values # by depth = 3 -> score 0.7761
#X = featuer_all # using 4 features -> score 0.9403

y = df_cleaned['species']
#print(y)
#print(X)
#print(features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20 % for the test
    random_state=42,     # Feste Zufallszahl für reproduzierbare Ergebnisse
    stratify=y           # Wichtig: Stellt sicher, dass das Verhältnis von Adelie/Gentoo in beiden Sets gleich ist
)

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print(f"Genauigkeit Decision Tree: {score:.4f}")
#print(f"Tiefe des Decision Tree: {pipe.named_steps['decisiontreeclassifier'].get_depth()}")
print(f"Anzahl der Blätter im Decision Tree: {pipe.named_steps['decisiontreeclassifier'].get_n_leaves()}")
#print(f"{pipe.get_params(deep=True)}")

# Visualisierung des Decision Tree
plt.figure(figsize=(12,8))

# visualize with two features
dt.plot_tree(pipe.named_steps['decisiontreeclassifier'], filled=True, feature_names=['bill_depth_mm', 'flipper_length_mm'], class_names=pipe.named_steps['decisiontreeclassifier'].classes_)   
# visualize with four features
#dt.plot_tree(pipe.named_steps['decisiontreeclassifier'], filled=True, feature_names=['bill_depth_mm', 'flipper_length_mm', 'bill_length_mm', 'body_mass_g'], class_names=pipe.named_steps['decisiontreeclassifier'].classes_) 
plt.title('Decision Tree for Penguin Species Classification')   
#plt.show()


## EX 4: Random Forest
# Ensemble Method: Random Forest
classifier = pipe_randomForest.fit(X_train, y_train)    #Train the Random Forest Classifier
score = pipe_randomForest.score(X_test, y_test)         #Genauigkeit Random Forest: 0.7910 same by rotated features

print(f"Genauigkeit Random Forest: {score:.4f}")

# Visualisierung eines Baumes im Random Forest
plt.figure(figsize=(12,8))
dt.plot_tree(pipe_randomForest.named_steps['randomforestclassifier'].estimators_[0], filled=True, feature_names=['bill_depth_mm', 'flipper_length_mm'], class_names=pipe_randomForest.named_steps['randomforestclassifier'].classes_)
plt.title('A Decision Tree from the Random Forest for Penguin Species Classification')


# Feature Importance
fig, ax = plt.subplots(figsize=(8, 6)) #create a new figure and axis for the bar plot
rf_estimator = pipe_randomForest.named_steps['randomforestclassifier']  #get the RandomForestClassifier from the pipeline
std = np.std([tree.feature_importances_ for tree in rf_estimator.estimators_], axis=0)  #calculate the standard deviation
importances = rf_estimator.feature_importances_ #get feature importances
feature_names = ['bill_depth_mm', 'flipper_length_mm']  #feature names for the bar plot
#feature_names = ['bill_depth_mm', 'flipper_length_mm', 'bill_length_mm', 'body_mass_g']

randomForestImportances = pd.Series(importances, index=feature_names)   #create a pandas Series from the numpy array for easy plotting

randomForestImportances.plot.bar(yerr=std, ax=ax)   #plot the feature importances with error bars
ax.set_title("Feature Importance (Random Forest)")
ax.set_ylabel("Wichtigkeit (Gini-Bedeutung)")
plt.xticks(rotation=0) # Beschriftungen horizontal halten
plt.tight_layout()


plt.show()


# Visualisierung der Trennung (Entscheidungsgrenzen)
# Erstelle eine neue Figure für diesen Plot
#plt.figure(figsize=(8, 6))
#ax = plt.gca()

# DecisionBoundaryDisplay plotten
# Plot the decision boundaries
#disp = DecisionBoundaryDisplay.from_estimator(
#classifier, X, response_method="predict",
#xlabel=features.feature_names[0], ylabel=features.feature_names[1], alpha=0.5
#)
#disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
#plt.show()