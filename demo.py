# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import mlflow
# import mlflow.pytorch
#
# # Load pre-trained VGG16 model
# model = torchvision.models.vgg16(pretrained=True)
#
# # Replace the classifier for fine-tuning (optional)
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 10)  # assuming 10 classes for the output
#
# # Move model to the device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # Prepare the dataset and data loader
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()
#
#             # Track metrics
#             running_loss += loss.item()
#
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         avg_loss = running_loss / len(train_loader)
#         accuracy = 100 * correct / total
#
#         # Log metrics and parameters to MLflow
#         mlflow.log_metric('loss', avg_loss, step=epoch)
#         mlflow.log_metric('accuracy', accuracy, step=epoch)
#
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
# # Start MLflow run
# with mlflow.start_run():
#     mlflow.log_param('batch_size', 64)
#     mlflow.log_param('epochs', 10)
#     mlflow.log_param('learning_rate', 0.001)
#
#     # Train the model and log the results
#     train_model(model, train_loader, criterion, optimizer, num_epochs=10)
#
#     # Log the trained model to MLflow
#     mlflow.pytorch.log_model(model, "vgg16_model")

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the experiment id
mlflow.set_experiment(experiment_id="4255919272347329")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
