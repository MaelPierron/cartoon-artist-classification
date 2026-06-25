import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Config
DATASET_5_PATH = "dataset5/train"   # dossier contenant un sous-dossier = la nouvelle classe
MODEL_PATH     = "model.pth"
NUM_CLASSES    = 4                  # nombre de classes d'origine (sur lesquelles la SVM a été entraînée)
SEUIL          = 0.6                # en dessous → rejeté comme "inconnu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exactement les mêmes transformations qu'à l'entraînement
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Chargement du 5ème dataset
dataset_5   = datasets.ImageFolder(DATASET_5_PATH, transform=transform)
loader_5    = DataLoader(dataset_5, batch_size=16, shuffle=False)
class_name  = dataset_5.classes[0]
print(f"Classe testée : {class_name} ({len(dataset_5)} images)")

# Chargement du modèle
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Modèle chargé.")

# Inférence
softmax     = nn.Softmax(dim=1)
all_proba   = []
all_pred_idx = []

with torch.no_grad():
    for images, _ in loader_5:
        images  = images.to(device)
        outputs = model(images)
        proba   = softmax(outputs)           # scores de confiance entre 0 et 1
        all_proba.append(proba.cpu().numpy())
        all_pred_idx.extend(proba.argmax(dim=1).cpu().numpy())

all_proba   = np.concatenate(all_proba, axis=0)
confidence  = all_proba.max(axis=1)          # score max par image

# Classes d'origine (ordre alphabétique, comme ImageFolder les lit)
# Remplace par les vrais noms de tes 4 classes si tu les connais
class_names = [f"classe_{i}" for i in range(NUM_CLASSES)]
# Exemple : class_names = ["chat", "chien", "oiseau", "poisson"]

# Résultats
n_total   = len(confidence)
n_rejetes = int((confidence < SEUIL).sum())
n_classes = n_total - n_rejetes

print(f"\n── Résultats sur le 5ème dataset ({class_name}) ──")
print(f"  Total images       : {n_total}")
print(f"  Rejetées (inconnu) : {n_rejetes}  ({100*n_rejetes/n_total:.1f}%)")
print(f"  Classées à tort    : {n_classes}  ({100*n_classes/n_total:.1f}%)")
print(f"\n  Confiance moyenne  : {confidence.mean():.3f}")
print(f"  Confiance min      : {confidence.min():.3f}")
print(f"  Confiance max      : {confidence.max():.3f}")

# Graphique 1 : distribution des scores de confiance
plt.figure(figsize=(8, 4))
plt.hist(confidence, bins=20, color='steelblue', edgecolor='white')
plt.axvline(SEUIL, color='red', linestyle='--', label=f"Seuil de rejet ({SEUIL})")
plt.xlabel("Score de confiance (softmax max)")
plt.ylabel("Nombre d'images")
plt.title(f"Distribution des scores de confiance — {class_name}")
plt.legend()
plt.tight_layout()
plt.show()

# Graphique 2 : à quelle classe ResNet assigne-t-il les images ?
counts = Counter(all_pred_idx)
labels_plot = [class_names[i] for i in counts.keys()]
values_plot = list(counts.values())

plt.figure(figsize=(7, 4))
plt.bar(labels_plot, values_plot, color='steelblue', edgecolor='white')
plt.xlabel("Classe prédite")
plt.ylabel("Nombre d'images")
plt.title(f"Classes prédites pour '{class_name}' (avant rejet)")
plt.tight_layout()
plt.show()