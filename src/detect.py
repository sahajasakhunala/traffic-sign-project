import cv2
import numpy as np

import torch
from PIL import Image
from torchvision import transforms

from model import TrafficSignCNN


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = TrafficSignCNN()

checkpoint = torch.load(
    "models/traffic_sign_cnn.pth",
    map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint["state_dict"])
model.eval()


# --------------------------------------------------
# Class names
# (expand later to all 43 classes)
# --------------------------------------------------
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


# --------------------------------------------------
# Image transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# --------------------------------------------------
# Load image
# --------------------------------------------------
image = cv2.imread("test.jpg")

if image is None:
    print("Could not find test.jpg")
    exit()


# --------------------------------------------------
# Convert to HSV
# --------------------------------------------------
hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)


# --------------------------------------------------
# Red masks
# --------------------------------------------------
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(
    hsv,
    lower_red1,
    upper_red1
)

mask2 = cv2.inRange(
    hsv,
    lower_red2,
    upper_red2
)

mask = mask1 + mask2


# --------------------------------------------------
# Find contours
# --------------------------------------------------
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# --------------------------------------------------
# Process detected regions
# --------------------------------------------------
for contour in contours:

    area = cv2.contourArea(contour)

    if area < 300:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        continue

    # Save crop
    cv2.imwrite(
        "detected_sign.jpg",
        crop
    )

    # Convert crop to PIL
    pil_image = Image.fromarray(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
    )

    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)

    # CNN prediction
    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = prediction.item()

    label = CLASS_NAMES.get(
        predicted_class,
        f"Class {predicted_class}"
    )

    confidence_text = (
        f"{label} "
        f"{confidence.item()*100:.1f}%"
    )

    # Draw rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Draw label
    cv2.putText(
        image,
        confidence_text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(
        f"Detected: {label} "
        f"({confidence.item()*100:.2f}%)"
    )


# --------------------------------------------------
# Show result
# --------------------------------------------------
cv2.imshow(
    "Traffic Sign Detection",
    image
)

cv2.waitKey(3000)

cv2.destroyAllWindows()

print("Detection complete.")