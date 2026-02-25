# con pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),              # Converts to [0,1]
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


images, labels = next(iter(train_loader))

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    img = images[i].permute(1,2,0)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc, val_acc = [], []
train_loss, val_loss = [], []

epochs = 10

for epoch in range(epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(correct / total)

    # Validation
    model.eval()
    correct, total, running_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss.append(running_loss / len(test_loader))
    val_acc.append(correct / total)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Acc: {train_acc[-1]:.4f} | "
          f"Val Acc: {val_acc[-1]:.4f}")


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.eval()
images, labels = next(iter(test_loader))
images, labels = images[:5].to(device), labels[:5].to(device)

with torch.no_grad():
    outputs = model(images)
    predictions = torch.softmax(outputs, dim=1)

def plot_image(i, predictions, true_labels, images, class_names):
    img = images[i].cpu().permute(1,2,0)
    pred_label = torch.argmax(predictions[i]).item()
    true_label = true_labels[i].item()

    color = 'blue' if pred_label == true_label else 'red'

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(
        f"Predicted: {class_names[pred_label]} "
        f"(True: {class_names[true_label]})",
        color=color
    )

plt.figure(figsize=(5,10))
for i in range(5):
    plt.subplot(5,1,i+1)
    plot_image(i, predictions, labels, images, class_names)

plt.tight_layout()
plt.show()