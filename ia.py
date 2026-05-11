import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

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

epochs = 10

for epoch in range(epochs):
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

    with torch.no_grad(): # on desactive le calcul des gradients pour la validation pour aller plus vite
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1) # on récupère la classe qui a un score maximal
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "model.pth")