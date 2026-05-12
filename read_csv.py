import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Chargement
df = pd.read_csv("features.csv")

X = df.drop(columns=["label"]).values
y = df["label"]

#  1. Normalisation
X = StandardScaler().fit_transform(X)

#  2. Sélection des meilleures features
selector = SelectKBest(score_func=f_classif, k=400)
X_reduced = selector.fit_transform(X, y)

#  3. PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_reduced)

# Labels numériques
labels = pd.factorize(y)[0]

#  Affichage
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    c=labels,
    cmap='tab10',
    alpha=0.7
)

plt.title("Projection PCA des meilleures features")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)
plt.show()