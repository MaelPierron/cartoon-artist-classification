import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

model_path = "model.pth"
image_path = "test.jpg"

# Doit être le même ordre que train_dataset.classes
class_names = ['Ghibli','One_piece','project_moon','steins_gate','tower_of_god']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations identiques à l'entraînement
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Charger le modèle
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Charger l'image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # ajouter batch dim
image = image.to(device)

# Prédiction
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

predicted_class = class_names[predicted.item()]
confidence = confidence.item()

print(f"Prédiction : {predicted_class}")
print(f"Confiance : {confidence:.2%}")