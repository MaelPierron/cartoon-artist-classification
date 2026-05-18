import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

X = df.drop(columns=["label"]).values # on enlève la colonne label pour ne garder que les features
y = df["label"] #ici on récupère les labels

X = StandardScaler().fit_transform(X) #on normalise les features (essentiel)

selector = SelectKBest(score_func=f_classif, k=400) # sélection des 400 meilleurs features (ceux qui maximisent les différences entre les classes)
X_reduced = selector.fit_transform(X, y)

X_2d = TSNE(n_components=2, perplexity=30).fit_transform(X_reduced) # application du TSNE
labels = pd.factorize(y)[0]
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=pd.factorize(y)[0])

plt.title("Projection TNSE des features")
plt.colorbar()
plt.show()