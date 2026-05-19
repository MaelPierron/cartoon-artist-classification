import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ── Chargement des objets sauvegardés ────────────────────────────────────────
scaler   = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
model    = joblib.load("model.pkl")
# ─────────────────────────────────────────────────────────────────────────────

# ── Chargement des features du 5ème dataset ──────────────────────────────────
df = pd.read_csv("features_5.csv")
X  = df.drop(columns=["label"]).values
y  = df["label"].values
# ─────────────────────────────────────────────────────────────────────────────

# ── Application du MÊME pipeline que l'entraînement ─────────────────────────
X = scaler.transform(X)      # transform() seulement, pas fit_transform() !
X = selector.transform(X)    # idem : on applique la sélection déjà apprise
# ─────────────────────────────────────────────────────────────────────────────

# ── Prédictions + scores de confiance ────────────────────────────────────────
y_pred   = model.predict(X)
y_proba  = model.predict_proba(X)        # probabilité pour chaque classe connue
confidence = y_proba.max(axis=1)         # score max = confiance de la prédiction
# ─────────────────────────────────────────────────────────────────────────────

# ── Seuil de rejet ───────────────────────────────────────────────────────────
SEUIL = 0.6   # en dessous → l'image est considérée "inconnue" / rejetée

decisions = []
for pred, conf in zip(y_pred, confidence):
    if conf < SEUIL:
        decisions.append("REJETÉ (inconnu)")
    else:
        decisions.append(f"Classé comme : {pred}")

# ── Résultats ─────────────────────────────────────────────────────────────────
n_total   = len(y)
n_rejetes = sum(1 for d in decisions if d == "REJETÉ (inconnu)")
n_classes = n_total - n_rejetes

print(f"\n── Résultats sur le 5ème dataset ({y[0]}) ──")
print(f"  Total images       : {n_total}")
print(f"  Rejetées (inconnu) : {n_rejetes}  ({100*n_rejetes/n_total:.1f}%)")
print(f"  Classées à tort    : {n_classes}  ({100*n_classes/n_total:.1f}%)")
print(f"\n  Confiance moyenne  : {confidence.mean():.3f}")
print(f"  Confiance min      : {confidence.min():.3f}")
print(f"  Confiance max      : {confidence.max():.3f}")

# ── Graphique 1 : distribution des scores de confiance ───────────────────────
plt.figure(figsize=(8, 4))
plt.hist(confidence, bins=20, color='steelblue', edgecolor='white')
plt.axvline(SEUIL, color='red', linestyle='--', label=f"Seuil de rejet ({SEUIL})")
plt.xlabel("Score de confiance (max proba)")
plt.ylabel("Nombre d'images")
plt.title(f"Distribution des scores de confiance — {y[0]}")
plt.legend()
plt.tight_layout()
plt.show()

# ── Graphique 2 : à quelle classe la SVM assigne-t-elle les images ? ─────────
from collections import Counter

counts = Counter(y_pred)
plt.figure(figsize=(7, 4))
plt.bar(counts.keys(), counts.values(), color='steelblue', edgecolor='white')
plt.xlabel("Classe prédite")
plt.ylabel("Nombre d'images")
plt.title(f"Classes prédites pour '{y[0]}' (avant rejet)")
plt.tight_layout()
plt.show()