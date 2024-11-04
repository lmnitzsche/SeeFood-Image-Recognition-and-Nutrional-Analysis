# Import necessary libraries
import pandas as pd
import numpy as np
from PIL import Image
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Other imports
import matplotlib.pyplot as plt
import seaborn as sns

# Load your DataFrame
df = pd.read_csv('data.csv')  # Replace with your actual file path

# Display basic information
print(df.head())
print(df['label'].value_counts())

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Custom Dataset Class
class FoodDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = sorted(df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'file']
        label = self.df.loc[idx, 'label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx

# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = FoodDataset(train_df, transform=train_transforms)
val_dataset = FoodDataset(val_df, transform=val_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [batch, 32, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 112, 112]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [batch, 64, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [batch, 128, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 128, 28, 28]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

# Initialize the model, criterion, optimizer
num_classes = len(train_dataset.labels)
simple_cnn_model = SimpleCNN(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(simple_cnn_model.parameters(), lr=0.001)

# Train the SimpleCNN model
print("Training SimpleCNN Model")
simple_cnn_model, train_losses, val_losses = train_model(simple_cnn_model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Evaluation function
def evaluate_model(model, dataloader, dataset):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.labels)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm

# Evaluate the SimpleCNN model
simple_cnn_acc, simple_cnn_report, simple_cnn_cm = evaluate_model(simple_cnn_model, val_loader, val_dataset)
print("SimpleCNN Validation Accuracy:", simple_cnn_acc)
print("Classification Report:")
print(simple_cnn_report)

# Transfer Learning with ResNet50
resnet50_model = models.resnet50(pretrained=True)

# Freeze the convolutional layers
for param in resnet50_model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_features = resnet50_model.fc.in_features
resnet50_model.fc = nn.Linear(num_features, num_classes)

resnet50_model = resnet50_model.to(device)

# Only parameters of the final layer are being optimized
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50_model.fc.parameters(), lr=0.001)

# Train the ResNet50 model
print("\nTraining ResNet50 Model with Transfer Learning")
resnet50_model, resnet50_train_losses, resnet50_val_losses = train_model(resnet50_model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Evaluate the ResNet50 model
resnet50_acc, resnet50_report, resnet50_cm = evaluate_model(resnet50_model, val_loader, val_dataset)
print("ResNet50 Validation Accuracy:", resnet50_acc)
print("Classification Report:")
print(resnet50_report)

# Feature extraction function
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.extend(label.numpy())
    return np.vstack(features), np.array(labels)

# Feature extractor model
feature_extractor = models.resnet50(pretrained=True)
# Remove the last classification layer
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove last fc layer
feature_extractor = feature_extractor.to(device)

# Extract features from training and validation sets
print("\nExtracting features for SVM and Random Forest")
train_features, train_labels = extract_features(feature_extractor, train_loader)
val_features, val_labels = extract_features(feature_extractor, val_loader)

# SVM classifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Standardize features before feeding into SVM
svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

print("\nTraining SVM Classifier")
svm_classifier.fit(train_features, train_labels)

# Predict on validation set
svm_preds = svm_classifier.predict(val_features)

# Evaluate SVM
svm_acc = accuracy_score(val_labels, svm_preds)
svm_report = classification_report(val_labels, svm_preds, target_names=train_dataset.labels)
svm_cm = confusion_matrix(val_labels, svm_preds)
print("SVM Validation Accuracy:", svm_acc)
print("Classification Report:")
print(svm_report)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTraining Random Forest Classifier")
rf_classifier.fit(train_features, train_labels)

# Predict on validation set
rf_preds = rf_classifier.predict(val_features)

# Evaluate Random Forest
rf_acc = accuracy_score(val_labels, rf_preds)
rf_report = classification_report(val_labels, rf_preds, target_names=train_dataset.labels)
rf_cm = confusion_matrix(val_labels, rf_preds)
print("Random Forest Validation Accuracy:", rf_acc)
print("Classification Report:")
print(rf_report)

# Collect metrics
metrics = {
    'Model': ['SimpleCNN', 'ResNet50', 'SVM', 'Random Forest'],
    'Accuracy': [simple_cnn_acc, resnet50_acc, svm_acc, rf_acc],
}

metrics_df = pd.DataFrame(metrics)
print("\nModel Comparison:")
print(metrics_df)

# Plot the accuracies
plt.figure(figsize=(8,6))
sns.barplot(x='Model', y='Accuracy', data=metrics_df)
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.show()

# Function to plot confusion matrices
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(simple_cnn_cm, train_dataset.labels, 'SimpleCNN Confusion Matrix')
plot_confusion_matrix(resnet50_cm, train_dataset.labels, 'ResNet50 Confusion Matrix')
plot_confusion_matrix(svm_cm, train_dataset.labels, 'SVM Confusion Matrix')
plot_confusion_matrix(rf_cm, train_dataset.labels, 'Random Forest Confusion Matrix')

# Save classification reports
with open('classification_reports.txt', 'w') as f:
    f.write("SimpleCNN Classification Report:\n")
    f.write(simple_cnn_report)
    f.write("\nResNet50 Classification Report:\n")
    f.write(resnet50_report)
    f.write("\nSVM Classification Report:\n")
    f.write(svm_report)
    f.write("\nRandom Forest Classification Report:\n")
    f.write(rf_report)
