#Tahniat Khayyam 577608
#For Deep Learning Class (Spring 2026)
#%%
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchsummary import summary

def to_one_hot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            preds = scores.argmax(dim=1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)
    model.train()
    return num_correct / num_samples


# Original Architecture
model1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 10),
    # nn.Softmax(dim=1)
)

# Increasing No. of Layers
model2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 128*2),
    nn.ReLU(),
    nn.Linear(128*2, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 10),
    # nn.Softmax(dim=1)
)

# Increase no. of neurons
model3 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128*4),
    nn.ReLU(),
    nn.Linear(128*4,64*4),
    nn.ReLU(),
    nn.Linear(64*4, 10),
    # nn.Softmax(dim=1)
)


# sigmoid
model4 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128*4),
    nn.Sigmoid(),
    nn.Linear(128*4,64*4),
    nn.Sigmoid(),
    nn.Linear(64*4, 10),
    # nn.Softmax(dim=1)
)


# tanh
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128*4),
    nn.Tanh(),
    nn.Linear(128*4,64*4),
    nn.Tanh(),
    nn.Linear(64*4, 10),
     
)



summary(model, input_size=(1, 28, 28)) # batch size 32
# loading the data

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) 
# Split into 80% train, 20% 
train_size = int(0.8 * len(train_dataset)) 
val_size = len(train_dataset) - train_size 
train_subset, val_subset = random_split(train_dataset, [train_size, val_size]) 
# DataLoaders for both
train_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=True) 
val_loader = DataLoader(dataset=val_subset, batch_size=32, shuffle=False) 

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

criterion =nn.BCEWithLogitsLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop 
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    epoch_loss = 0
    
    # Training
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.reshape(data.shape[0], -1)
        targets = to_one_hot(targets, num_classes=10)  # one-hot labels
        
        scores = model(data)
        loss = criterion(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        epoch_loss += loss.item()
    
    # Average train loss
    train_losses.append(epoch_loss / len(train_loader))
    
    # Train accuracy
    train_acc = check_accuracy(train_loader, model)
    train_accuracies.append(train_acc)
    
    # Validation
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.reshape(data.shape[0], -1)
            targets = to_one_hot(targets, num_classes=10)  # one-hot labels here too!
            scores = model(data)
            loss = criterion(scores, targets)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))
    val_acc = check_accuracy(val_loader, model)
    val_accuracies.append(val_acc)

    
    print(f"Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")



# plotting loss and accuracy

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

# Loss
plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# testing accuracy
test_acc = check_accuracy(test_loader, model)
print(f"Test Acc: {test_acc:.3f}")
