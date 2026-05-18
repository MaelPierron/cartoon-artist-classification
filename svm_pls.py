import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "features.csv")

df = pd.read_csv(csv_path)

X = df.drop(columns=["label"]).values
y = df["label"].values

# Encodage des labels
from sklearn.preprocessing import LabelEncoder

# Labels simples pour SVM
label_encoder = LabelEncoder()
y_svm = label_encoder.fit_transform(y)

# One-hot pour PLS
onehot = OneHotEncoder(sparse_output=False)
y_pls = onehot.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train_svm, y_test_svm, y_train_pls, y_test_pls = train_test_split(
    X,
    y_svm,
    y_pls,
    test_size=0.2,
    random_state=42,
    stratify=y_svm
)

# Normalisation
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_acc = 0
best_n = 0

# for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]:

#     pls = PLSRegression(n_components=n)

#     X_train_pls = pls.fit_transform(X_train, y_train_pls)[0]
#     X_test_pls = pls.transform(X_test)

#     svm = SVC(kernel='rbf', C=10)

#     svm.fit(X_train_pls, y_train_svm)

#     y_pred = svm.predict(X_test_pls)

#     acc = accuracy_score(y_test_svm, y_pred)

#     print(f"{n} composantes -> {acc:.4f}")

#     if acc > best_acc:
#         best_acc = acc
#         best_n = n

# print("\nMeilleur résultat :")
# print(best_n, "composantes")
# print("Accuracy :", best_acc)

pls = PLSRegression(n_components=15)

X_train_pls = pls.fit_transform(X_train, y_train_pls)[0]
X_test_pls = pls.transform(X_test)

svm = SVC(kernel='rbf', C=10)

svm.fit(X_train_pls, y_train_svm)

y_pred = svm.predict(X_test_pls)

# t-SNE sur les features PLS (test set)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne = tsne.fit_transform(X_test_pls)

# affichage
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y_test_svm,
    cmap='tab10',
    alpha=0.7
)

# légende propre
handles, _ = scatter.legend_elements()
labels = label_encoder.classes_

plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1))

plt.title("t-SNE après PLS (test set)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.show()

acc = accuracy_score(y_test_svm, y_pred)

cm = confusion_matrix(y_test_svm, y_pred)

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_percent,
    display_labels=label_encoder.classes_
)

disp.plot(cmap='Blues', values_format=".1f")
plt.title("Confusion Matrix PLS + SVM (%)")
plt.xticks(rotation=45)
plt.show()

