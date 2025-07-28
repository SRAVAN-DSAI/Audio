import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)
num_classes = 10  # UrbanSound8K has 10 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final layer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (assume dataloader is provided)
def train_model(dataloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder('spectrograms/', transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_model(dataloader, epochs=5)