import torch
from PIL import Image
from torchvision import transforms

from model import TrafficSignCNN

# Class names (partial list for now)
CLASS_NAMES = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Passing",
    10: "No Passing Trucks",
    11: "Right of Way",
    12: "Priority Road",
    13: "Yield",
    14: "Stop"
}

# Load trained model
model = TrafficSignCNN()

checkpoint = torch.load(
    "models/traffic_sign_cnn.pth",
    map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint["state_dict"])

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Change this path to test different images
image_path = "detected_sign.jpg"

# Load image
image = Image.open(image_path).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

# Prediction
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

predicted_class = prediction.item()

print("Predicted Class:", predicted_class)

print(
    "Sign:",
    CLASS_NAMES.get(
        predicted_class,
        "Unknown Sign"
    )
)

print(
    "Confidence:",
    round(confidence.item() * 100, 2),
    "%"
)