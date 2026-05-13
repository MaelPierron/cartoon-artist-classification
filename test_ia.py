import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

model_path = "model.pth"
test_folder = "tests_squares"


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

# Extensions d'images supportées
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Parcourir toutes les images du dossier
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(image_extensions)]

print(f"{len(image_files)} image(s) trouvée(s) dans '{test_folder}'\n")

for filename in sorted(image_files):
    image_path = os.path.join(test_folder, filename)

    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = confidence.item()

    print(f"{filename:<30} → {predicted_class:<15} ({confidence:.2%})")