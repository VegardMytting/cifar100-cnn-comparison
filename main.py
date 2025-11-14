import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np
from InquirerPy import inquirer
from rich.console import Console
from tqdm import tqdm
import statistics

EPOCHS = 50
RUNS = 3

console = Console()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[#e5c07b]![/#e5c07b] Using device: [#61afef]{device}[/#61afef]")

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

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

trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

def resnet18():
  from torchvision.models import resnet18, ResNet18_Weights
  model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def resnet50():
  from torchvision.models import resnet50, ResNet50_Weights
  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def efficientnetb0():
  from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
  model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
  return model

def mobilenetv3_large():
  from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
  model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
  model.classifier[3] = nn.Linear(model.classifier[3].in_features, 100)
  return model

def shufflenetv2():
  from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
  model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def resnext50():
  from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
  model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def convnext_tiny():
  from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
  model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
  model.classifier[2] = nn.Linear(model.classifier[2].in_features, 100)
  return model

def vit_b16():
  from torchvision.models import vit_b_16, ViT_B_16_Weights
  model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
  model.heads.head = nn.Linear(model.heads.head.in_features, 100)
  return model

def wideresnet2810():
  from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
  model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def customcnn():
  class CustomCNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
      self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
      self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.fc1 = nn.Linear(256 * 4 * 4, 512)
      self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      return self.fc2(x)

  return CustomCNN()

exporters = {
  "Custom CNN": customcnn,
  "ResNet18": resnet18,
  "ResNet50": resnet50,
  "ResNeXt50-32x4d": resnext50,
  "EfficientNet-B0": efficientnetb0,
  "MobileNetV3-Large": mobilenetv3_large,
  "ShuffleNetV2-1.0": shufflenetv2,
  "ConvNeXt-Tiny": convnext_tiny,
  "ViT-B/16": vit_b16,
  "WideResNet-50-2": wideresnet2810,
}

model_choice = inquirer.select(
  message="Select a model architecture:",
  choices=list(exporters.keys()),
).execute()

def build_model(name):
  return exporters[name]()
  
accuracies = []
console.print(f"[bold cyan]\nRunning {RUNS} training runs for {model_choice}...[/bold cyan]")

for run in tqdm(range(RUNS), desc="Runs", position=0, leave=False):
  set_seed(run)
  model = build_model(model_choice).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0005)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

  for epoch in tqdm(range(EPOCHS), desc=f"Training (Run {run+1})", position=1, leave=False):
    model.train()
    running_loss = 0.0
      
    with tqdm(trainloader, desc=f"Epoch {epoch+1}", position=2, leave=False) as pbar:
      for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss/len(trainloader):.4f}")

    scheduler.step()

  correct = 0
  total = 0
  model.eval()

  with torch.no_grad():
    with tqdm(testloader, desc=f"Testing (Run {run+1})", position=1, leave=False) as pbar:
      for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")

  accuracy = 100 * correct / total
  accuracies.append(accuracy)

mean_acc = statistics.mean(accuracies)
std_acc = statistics.stdev(accuracies)

console.print("\n[bold cyan]===== FINAL SUMMARY =====[/bold cyan]")
for i, acc in enumerate(accuracies):
  console.print(f"Run {i+1}: {acc:.2f}%")

console.print(f"\n[bold green]Mean accuracy: {mean_acc:.2f}%[/bold green]")
console.print(f"[bold magenta]Std deviation: {std_acc:.2f}[/bold magenta]")
