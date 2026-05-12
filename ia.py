import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(), # transformation de l'image en matrice exploitable par le système
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]) # on normalise entre -1 et 1 pour faciliter l'apprentissage
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

model = models.resnet18(pretrained=True) # on charge le modèle préentraîné resnet18

model.fc = nn.Linear(model.fc.in_features, num_classes) #on adapte le modèle à notre nombre de classes
model = model.to(device)

criterion = nn.CrossEntropyLoss() # choix, de la loss, EntropyLoss classique pour classification d'images
optimizer = optim.Adam(model.parameters(), lr=1e-4)


previous_loss = 0
current_loss = 1000
thresh = 0.2
epoch = 0
while np.abs(previous_loss-current_loss)>thresh:
    
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    correct = 0
    total = 0

    # stockage des labels
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 🔥 sauvegarde
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    epoch+=1
    previous_loss = current_loss
    current_loss = total_loss

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)

# Conversion en %
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Affichage
fig, ax = plt.subplots(figsize=(8,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_percent,
    display_labels=train_dataset.classes
)

disp.plot(
    cmap='Blues',
    ax=ax,
    values_format=".1f"
)

plt.xticks(rotation=45)
plt.title("Confusion Matrix CNN (%)")
plt.show()
torch.save(model.state_dict(), "model.pth")