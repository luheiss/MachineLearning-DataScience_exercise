import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from palmerpenguins import load_penguins  # richtige Import-Schreibweise

# Datensatz laden und säubern
peng = load_penguins()
cleaned_peng = peng.dropna()
sns.pairplot(cleaned_peng, hue='species')
plt.show()
# print(cleaned_peng.head())

features = ['bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_length_mm']
X = cleaned_peng[features].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = cleaned_peng['species'].values

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', palette='Set1')
plt.title('PCA der Palmer Penguins (4 Merkmale → 2 Komponenten)')
plt.show()
