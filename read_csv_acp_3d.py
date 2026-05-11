import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Chargement
df = pd.read_csv("features.csv")

X = df.drop(columns=["label"]).values
y = df["label"]

# 🔥 1. Normalisation
X = StandardScaler().fit_transform(X)

# 🔥 2. Sélection des meilleures features
selector = SelectKBest(score_func=f_classif, k=400)
X_reduced = selector.fit_transform(X, y)

# 🔥 3. PCA
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X_reduced)

# Labels numériques
labels = pd.factorize(y)[0]

# 🔥 4. Plot 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_3d[:,0],
    X_3d[:,1],
    X_3d[:,2],
    c=labels,
    cmap='tab10',
    alpha=0.7
)

ax.set_title("Projection PCA 3D des features")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.colorbar(scatter)
plt.show()