from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="data/GTSRB/GTSRB",
    transform=transform
)

print("Images:", len(dataset))
print("Classes:", len(dataset.classes))

for i in range(5):
    image, label = dataset[i]

    plt.figure()
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Class: {label}")
    plt.axis("off")

plt.show()