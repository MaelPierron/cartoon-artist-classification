import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

#  Chargement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "features.csv")

df = pd.read_csv(csv_path)

X = df.drop(columns=["label"]).values
y = df["label"].values

#  Encodage des labels
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#  Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

#  Normalisation
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #  PLS
# pls = PLSRegression(n_components=20)

# X_train_pls = pls.fit_transform(X_train, y_train)[0]
# X_test_pls = pls.transform(X_test)

# #  SVM
# svm = SVC(kernel='rbf', C=10)

# svm.fit(X_train_pls, y_train)

# #  Prédictions
# y_pred = svm.predict(X_test_pls)

# #  Accuracy
# acc = accuracy_score(y_test, y_pred)

# print("Accuracy :", acc)

best_acc = 0
best_n = 0

for n in [2, 5, 7, 10, 20, 30, 50]:

    pls = PLSRegression(n_components=n)

    X_train_pls = pls.fit_transform(X_train, y_train)[0]
    X_test_pls = pls.transform(X_test)

    svm = SVC(kernel='rbf', C=10)

    svm.fit(X_train_pls, y_train)

    y_pred = svm.predict(X_test_pls)

    acc = accuracy_score(y_test, y_pred)

    print(f"{n} composantes -> {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_n = n

print("\nMeilleur résultat :")
print(best_n, "composantes")
print("Accuracy :", best_acc)