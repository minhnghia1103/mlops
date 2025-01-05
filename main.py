from src.train import train_model
from src.model import model
import torchvision
import torchvision.transforms as transform
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import mlflow.pytorch
import config


# Tải bộ dữ liệu CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Hàm lọc dữ liệu chỉ lấy 3 lớp đầu tiên (0: Máy bay, 1: Xe hơi, 2: Chim)
def filter_classes(dataset, classes=[0, 1, 2]):
    indices = [i for i, label in enumerate(dataset.targets) if label in classes]
    return Subset(dataset, indices)

# Lọc dữ liệu chỉ lấy 3 lớp đầu tiên
filtered_train_dataset = filter_classes(train_dataset)
filtered_test_dataset = filter_classes(test_dataset)

train_size = int(0.8 * len(filtered_train_dataset))  # 80% dữ liệu cho training
val_size = len(filtered_train_dataset) - train_size  # 20% dữ liệu cho validation

train_subset, val_subset = random_split(filtered_train_dataset, [train_size, val_size])

# Tạo DataLoader cho từng tập
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(filtered_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('epochs', NUM_EPOCHS)
    mlflow.log_param('learning_rate', LEARNING_RATE)

    # Train the model and log the results
    train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Log the trained model to MLflow
    mlflow.pytorch.log_model(model, "vgg16_model")

