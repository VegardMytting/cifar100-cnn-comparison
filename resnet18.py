import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4865, 0.4409),
                       (0.2673, 0.2564, 0.2761)),
])

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4865, 0.4409),
                       (0.2673, 0.2564, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 100)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(40):
  print(f"Epoch {epoch+1}/40")
  
  model.train()
  running_loss = 0.0
  
  for imgs, labels in trainloader:
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
      
  scheduler.step()
  print(f"Loss: {running_loss/len(trainloader):.4f}")

correct = 0
total = 0
model.eval()

with torch.no_grad():
  for imgs, labels in testloader:
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total)
