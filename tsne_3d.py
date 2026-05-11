import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

X = df.drop(columns=["label"]).values
y = df["label"]

# 1. normalisation
X = StandardScaler().fit_transform(X)

# 2. sélection des features
selector = SelectKBest(score_func=f_classif, k=400)
X_reduced = selector.fit_transform(X, y)

# 3. t-SNE en 3D
X_3d = TSNE(n_components=3, perplexity=30, random_state=42).fit_transform(X_reduced)

labels = pd.factorize(y)[0]

# 4. plot 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X_3d[:,0],
    X_3d[:,1],
    X_3d[:,2],
    c=labels,
    cmap='tab10',
    alpha=0.7
)

ax.set_title("t-SNE 3D des features")
plt.colorbar(sc)
plt.show()