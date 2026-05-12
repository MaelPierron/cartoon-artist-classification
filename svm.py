
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay



df = pd.read_csv("features.csv")

# 🔥 Features / labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# 🔥 1. Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔥 2. Sélection des meilleures features
selector = SelectKBest(score_func=f_classif, k=400)
X = selector.fit_transform(X, y)

print("Shape après sélection :", X.shape)

# 🔥 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 🔥 4. SVM
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

print("Entraînement du SVM...")
model.fit(X_train, y_train)

# 🔥 5. Prédictions
y_pred = model.predict(X_test)

# 🔥 6. Évaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

# print("\nClassification report :")
# print(classification_report(y_test, y_pred))

# print("\nConfusion matrix :")
# print(confusion_matrix(y_test, y_pred))

# 🔥 Calcul confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 🔥 Conversion en pourcentage
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 🔥 Affichage
fig, ax = plt.subplots(figsize=(8,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_percent,
    display_labels=model.classes_
)

disp.plot(
    cmap='Blues',
    ax=ax,
    values_format=".1f"
)

plt.title("Confusion Matrix (%)")
plt.show()