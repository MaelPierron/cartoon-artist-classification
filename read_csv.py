import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

X = df.drop(columns=["label"]).values
y = df["label"]

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
labels = pd.factorize(y)[0]
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=pd.factorize(y)[0], alpha=0.6)

plt.title("Projection PCA des features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()