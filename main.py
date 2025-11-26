import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statistics
import torchvision
import torchvision.transforms as transforms
import random
from tqdm import tqdm
from InquirerPy import inquirer
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

console = Console()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[#e5c07b]![/#e5c07b] Using device: [#61afef]{device}[/#61afef]")

RUNS = int(inquirer.number(
  message="Select number of Runs:",
  default=3,
  min_allowed=2
).execute())

EPOCHS = int(inquirer.number(
  message="Select number of Epochs:",
  default=50,
  min_allowed=1
).execute())

def set_seed(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
activation_function_registry = {
  "ReLU": nn.ReLU,
  "LeakyReLU": nn.LeakyReLU,
  "GELU": nn.GELU,
  "Sigmoid": nn.Sigmoid,
  "Tanh": nn.Tanh,
  "Softmax": nn.Softmax(dim=1),
  "Mish": nn.Mish
}

def custom_cnn(activation_function = "ReLU"):
  class CustomCNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.activation_function = activation_function
      
      self.features = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        activation_function_registry[activation_function](),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        activation_function_registry[activation_function](),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        activation_function_registry[activation_function](),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        activation_function_registry[activation_function](),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        activation_function_registry[activation_function](),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        activation_function_registry[activation_function](),
        nn.MaxPool2d(2, 2),
        
        nn.AdaptiveAvgPool2d((1, 1))
      )
      
      self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(256, 512),
        activation_function_registry[activation_function](),
        nn.Dropout(0.5),
        nn.Linear(512, 100)
      )
        
    def forward(self, x):
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x
      
  return CustomCNN()

def resnet18(use_pretrained_weights: bool = True):
  from torchvision.models import resnet18, ResNet18_Weights
  model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if use_pretrained_weights else resnet18()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def resnet34(use_pretrained_weights: bool = True):
  from torchvision.models import resnet34, ResNet34_Weights
  model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) if use_pretrained_weights else resnet34()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def resnet50(use_pretrained_weights: bool = True):
  from torchvision.models import resnet50, ResNet50_Weights
  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if use_pretrained_weights else resnet50()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def resnext50(use_pretrained_weights: bool = True):
  from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
  model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1) if use_pretrained_weights else resnext50_32x4d()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def efficientnetb0(use_pretrained_weights: bool = True):
  from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
  model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1) if use_pretrained_weights else efficientnet_b0()
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
  return model

def efficientnet_b2(use_pretrained_weights: bool = True):
  from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
  model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1) if use_pretrained_weights else efficientnet_b2()
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
  return model

def mobilenetv3_large(use_pretrained_weights: bool = True):
  from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
  model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1) if use_pretrained_weights else mobilenet_v3_large()
  model.classifier[3] = nn.Linear(model.classifier[3].in_features, 100)
  return model

def shufflenetv2(use_pretrained_weights: bool = True):
  from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
  model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1) if use_pretrained_weights else shufflenet_v2_x1_0()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def convnext_tiny(use_pretrained_weights: bool = True):
  from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
  model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if use_pretrained_weights else convnext_tiny()
  model.classifier[2] = nn.Linear(model.classifier[2].in_features, 100)
  return model

def wideresnet50_2(use_pretrained_weights: bool = True):
  from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
  model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1) if use_pretrained_weights else wide_resnet50_2()
  model.fc = nn.Linear(model.fc.in_features, 100)
  return model

def densenet121(use_pretrained_weights: bool = True):
  from torchvision.models import densenet121, DenseNet121_Weights
  model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1) if use_pretrained_weights else densenet121()
  model.classifier = nn.Linear(model.classifier.in_features, 100)
  return model

cnn_registry = {
  "Custom CNN": custom_cnn,
  "ResNet18": resnet18,
  # "ResNet34": resnet34,
  "ResNet50": resnet50,
  "ResNeXt50-32x4d": resnext50,
  "EfficientNet-B0": efficientnetb0,
  # "EfficientNet-B2": efficientnet_b2,
  "MobileNetV3-Large": mobilenetv3_large,
  "ShuffleNetV2-1.0": shufflenetv2,
  "ConvNeXt-Tiny": convnext_tiny,
  "WideResNet-50-2": wideresnet50_2,
  "DenseNet121": densenet121,
}

baseline_registry = {
  "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=66, verbose=0),
  "Decision Tree": lambda: DecisionTreeClassifier(max_depth=20, random_state=66),
  "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=20, random_state=66, n_jobs=-1, verbose=0),
  "SVM (Linear)": lambda: SVC(kernel='linear', random_state=66, verbose=False),
  "SVM (RBF)": lambda: SVC(kernel='rbf', random_state=66, verbose=False),
  "K-Nearest Neighbors (K=3)": lambda: KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
  "K-Nearest Neighbors (K=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
  "K-Nearest Neighbors (K=10)": lambda: KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
  "Naive Bayes (Gaussian)": lambda: GaussianNB(),
  "MLP (2 hidden layers)": lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=66, verbose=0),
}

preprocessing_mode = inquirer.select(
  message="Select data preprocessing:",
  choices=[
    "Augmentation & Normalize",
    "Augmentation",
    "Normalize",
    "No preprocessing",
  ],
).execute()

def get_transforms(mode):
  if mode == "Augmentation & Normalize":
    transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
      )
    ])
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
      )
    ])
      
  elif mode == "Augmentation":
    transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
    ])
  
  elif mode == "Normalize":
    transform_train = transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
      )
    ])
    
  elif mode == "No preprocessing":
    transform_train = transform_test = transforms.Compose([
      transforms.ToTensor()
    ])
    
  else:
    raise ValueError(f"Unknown preprocessing mode: {mode}")
  
  return transform_train, transform_test

def init_data(mode="Augmentation & Normalize"):
  transform_train, transform_test = get_transforms(mode)

  trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  return trainset, trainloader, testset, testloader

trainset, trainloader, testset, testloader = init_data(mode=preprocessing_mode)

def train_cnn(model_name, optimizer_name, learning_rate, use_transfer_learning = False, cnn_activation_function = None):
  accuracies = []
  
  transfer_status = "with transfer learning" if use_transfer_learning else "without transfer learning"
  if model_name != "Custom CNN":
    console.print(f"[#e5c07b]![/#e5c07b] Running [#61afef]{RUNS}[/#61afef] training rounds for [#61afef]{model_name}[/#61afef] {transfer_status}, and with the [#61afef]{optimizer_name}[/#61afef] Optimizer ([#61afef]lr[/#61afef]=[#61afef]{learning_rate}[/#61afef])")
  else:
    console.print(f"[#e5c07b]![/#e5c07b] Running [#61afef]{RUNS}[/#61afef] training rounds for [#61afef]{model_name}[/#61afef] with the [#61afef]{cnn_activation_function}[/#61afef] activation function, and with the [#61afef]{optimizer_name}[/#61afef] Optimizer ([#61afef]lr[/#61afef]=[#61afef]{learning_rate}[/#61afef])")
  for run in tqdm(range(RUNS), desc="Runs", position=0, leave=False):
    set_seed(run)
    
    if model_name != "Custom CNN":
      model = cnn_registry[model_name](use_pretrained_weights=use_transfer_learning).to(device)
    else:
      model = cnn_registry[model_name](activation_function=cnn_activation_function).to(device)
      
    criterion = nn.CrossEntropyLoss()
      
    match optimizer_name:
      case "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      case "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
      case "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
      case "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
      case _:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")
    
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
  
  console.print(f"\n[#e5c07b]![/#e5c07b] Accuracies: \t\t{accuracies}")
  console.print(f"[#e5c07b]![/#e5c07b] Mean accuracy: \t{mean_acc:.2f} %")
  console.print(f"[#e5c07b]![/#e5c07b] Std deviation: \t± {std_acc:.2f}")

def train_baseline(model_name, pca_components):
  console.print("[#e5c07b]![/#e5c07b] Preparing data for ML baseline")
  
  def prepare_data(dataset, max_samples=None):
    if max_samples:
      indices = np.random.choice(len(dataset), max_samples, replace=False)
      data = [dataset[i] for i in tqdm(indices, desc="Loading samples", leave=False)]
    else:
      data = [dataset[i] for i in tqdm(range(len(dataset)), desc="Loading samples", leave=False)]
      
    x = np.array([img.numpy().flatten() for img, _ in data])
    y = np.array([label for _, label in data])
    
    return x, y
  
  x_train, y_train = prepare_data(trainset, max_samples=None)
  x_test, y_test = prepare_data(testset)
  
  console.print(f"[#e5c07b]![/#e5c07b] Training data shape: {x_train.shape}")
  console.print(f"[#e5c07b]![/#e5c07b] Test data shape: {x_test.shape}")
  
  console.print(f"[#e5c07b]![/#e5c07b] Standardizing features and applying PCA ([#61afef]n_components[/#61afef]=[#61afef]{pca_components}[/#61afef])")
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(x_train)
  X_test_scaled = scaler.transform(x_test)
  
  pca = PCA(n_components=pca_components, random_state=66)
  X_train_pca = pca.fit_transform(X_train_scaled)
  X_test_pca = pca.transform(X_test_scaled)
  
  if isinstance(pca_components, float):
    console.print(f"[#e5c07b]![/#e5c07b] Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    console.print(f"[#e5c07b]![/#e5c07b] Number of components used: {pca.n_components_}")
  else:
    console.print(f"[#e5c07b]![/#e5c07b] Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
  accuracies = []
  
  console.print(f"[#e5c07b]![/#e5c07b] Running [#61afef]{RUNS}[/#61afef] training rounds for [#61afef]{model_name}[/#61afef]")
  for run in tqdm(range(RUNS), desc="Runs", position=0, leave=False):
    set_seed(run)

    model = baseline_registry[model_name]()
    model.random_state = run
    
    with tqdm(total=1, desc=f"Training (Run {run+1})", position=1, leave=False) as pbar:
      model.fit(X_train_pca, y_train)
      pbar.update(1)
    
    with tqdm(total=1, desc=f"Testing (Run {run+1})", position=1, leave=False) as pbar:
      test_acc = model.score(X_test_pca, y_test) * 100
      pbar.update(1)
    
    accuracies.append(test_acc)
    
  mean_acc = statistics.mean(accuracies)
  std_acc = statistics.stdev(accuracies)
  
  console.print(f"\n[#e5c07b]![/#e5c07b] Model Name: \t{model_name}")
  console.print(f"[#e5c07b]![/#e5c07b] Accuracies: \t\t{accuracies}")
  console.print(f"[#e5c07b]![/#e5c07b] Mean accuracy: \t{mean_acc:.2f} %")
  console.print(f"[#e5c07b]![/#e5c07b] Std deviation: \t± {std_acc:.2f}")

model_type = inquirer.select(
  message="Select Model Type:",
  choices=["Deep Learning (CNNs)", "ML Baselines"],
).execute()

if model_type == "Deep Learning (CNNs)":
  run_all = inquirer.confirm(
    message="Run All Models?",
    default=False
  ).execute()
  
  if not run_all:
    model_choice = inquirer.select(
      message="Select a CNN Architecture:",
      choices=list(cnn_registry.keys()),
    ).execute()
  
  if run_all or model_choice != "Custom CNN":
    use_transfer_learning = inquirer.confirm(
      message="Use Transfer Learning (Pretrained ImageNet Weights)?",
      default=True
    ).execute()
    
  cnn_activation_function = None
    
  if run_all or model_choice == "Custom CNN":
    cnn_activation_function = inquirer.select(
      message="Select Activation Function for the Custom CNN:",
      choices=activation_function_registry.keys()
    ).execute()
  
  optimizer_choice = inquirer.select(
    message="Select an Optimizer:",
    choices=["Adam", "SGD", "AdamW", "RMSprop"],
  ).execute()
  
  learning_rate = float(inquirer.text(
    message="Select Learning Rate:",
    default="0.0005"
  ).execute())
  
  if run_all:
    for model_name in tqdm(cnn_registry.keys(), desc="models", leave=False):
      train_cnn(model_name, optimizer_choice, learning_rate, use_transfer_learning = use_transfer_learning, cnn_activation_function = cnn_activation_function)
  else:
    train_cnn(model_choice, optimizer_choice, learning_rate, use_transfer_learning = use_transfer_learning, cnn_activation_function = cnn_activation_function)
else:
  model_choice = inquirer.select(
    message="Select a Baseline Model:",
    choices=list(baseline_registry.keys()),
  ).execute()

  pca_components = float(inquirer.text(
    message="Select PCA n_components:",
    default="0.95"
  ).execute())
  
  train_baseline(model_choice, pca_components)

# def customcnn():
#   class CustomCNN(nn.Module):
#     def __init__(self):
#       super().__init__()
#       self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
#       self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#       self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
#       self.pool = nn.MaxPool2d(2, 2)
#       self.fc1 = nn.Linear(256 * 4 * 4, 512)
#       self.fc2 = nn.Linear(512, 100)

#     def forward(self, x):
#       x = self.pool(F.relu(self.conv1(x)))
#       x = self.pool(F.relu(self.conv2(x)))
#       x = self.pool(F.relu(self.conv3(x)))
#       x = x.view(x.size(0), -1)
#       x = F.relu(self.fc1(x))
#       return self.fc2(x)

#   return CustomCNN()