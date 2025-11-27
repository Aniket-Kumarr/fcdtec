#!/usr/bin/env python3
"""
Train Rock-Paper-Scissors Gesture Recognition Model
Uses MobileNetV2 for transfer learning
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Check if data exists
if not os.path.exists("data"):
    print("Error: 'data' folder not found!")
    print("Please run collect_data.py first to collect training images.")
    exit(1)

# Check if we have data
classes = ["rock", "paper", "scissors"]
for cls in classes:
    cls_path = f"data/{cls}"
    if not os.path.exists(cls_path):
        print(f"Warning: {cls_path} does not exist!")
    else:
        count = len([f for f in os.listdir(cls_path) if f.endswith('.jpg')])
        print(f"{cls}: {count} images")

# Data augmentation and transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading dataset...")
train_data = datasets.ImageFolder("data/", transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)

print(f"Total images: {len(train_data)}")
print(f"Classes: {train_data.classes}")

# Create model (MobileNetV2 with pretrained weights)
print("\nCreating model...")
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Modify classifier for 3 classes (rock, paper, scissors)
model.classifier[1] = nn.Linear(model.last_channel, 3)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
print("\nStarting training...")
print("=" * 50)

# Adjust epochs based on dataset size
num_images = len(train_data)
if num_images < 50:
    num_epochs = 20  # More epochs for small datasets
    print(f"⚠️  Warning: Small dataset ({num_images} images). For best results, collect 200+ images per class.")
    print("   Training with more epochs to compensate...")
elif num_images < 200:
    num_epochs = 15
    print(f"⚠️  Dataset size: {num_images} images. Consider collecting more for better accuracy.")
else:
    num_epochs = 10
    print(f"✓ Good dataset size: {num_images} images")

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

print("=" * 50)
print("Training complete!")

# Save model
model_path = 'rps_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")

# Save class names for inference
import json
with open('class_names.json', 'w') as f:
    json.dump(train_data.classes, f)
print("Class names saved to: class_names.json")

