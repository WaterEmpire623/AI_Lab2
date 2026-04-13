import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

class AugmentedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment_factor=1, augment_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment_factor = augment_factor
        self.augment_transform = augment_transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        image_idx = idx // (1 + self.augment_factor)
        img_path = os.path.join(self.image_dir, self.images[image_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[image_idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0) # Load as grayscale (0-15) [cite: 14]
        
        # Determine if this is an augmented sample
        is_augmented = idx % (1 + self.augment_factor) != 0
        
        if is_augmented and self.augment_transform:
            # Note: For segmentation, augmentation must apply to BOTH img and mask
            # Simplified here for clarity; libraries like Albumentations are better for this
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)
            
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

def prepare_dataloaders(image_dir, mask_dir, batch_size=16, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AugmentedSegmentationDataset(image_dir, mask_dir, transform=transform)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleUNet, self).__init__()
        # Encoder: Pretrained ResNet18 [cite: 608]
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # Decoder [cite: 612]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Resize back to original UAV image size (600, 800) [cite: 635]
        return nn.functional.interpolate(x, size=(600, 800), mode="bilinear")
    
class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = nn.functional.softmax(inputs, dim=1)
        total_loss = 0
        for i in range(self.num_classes):
            if i == 0: continue # Skip background [cite: 570]
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = (targets == i).float().view(-1)
            intersection = (input_flat * target_flat).sum()
            union = input_flat.sum() + target_flat.sum()
            total_loss += 1 - ((2. * intersection + 1e-5) / (union + 1e-5))
        return total_loss / self.num_classes

def calculate_iou(outputs, masks, num_classes):
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(1, num_classes): # Skip background [cite: 570]
        pred = (outputs == cls)
        target = (masks == cls)
        intersection = (pred & target).sum().item()
        union = (pred | target).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 16  # Based on HW2 spec 
model = SimpleUNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training
epochs = 20
history = {'train_loss': [], 'val_iou': []}

for epoch in range(epochs):
    model.train()
    train_loader, val_loader = prepare_dataloaders("./train/imgs", "./train/masks")
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
    # Validation [cite: 751]
    model.eval()
    ious = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            ious.append(calculate_iou(outputs, masks, num_classes))
    
    print(f"Mean IoU: {np.mean(ious):.4f}")
    history['val_iou'].append(np.mean(ious))

# Save weights [cite: 777]
torch.save(model.state_dict(), "best_model.pth")

# ===== Plotting Training and Validation Loss =====
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss_history, label="Training Loss", color='blue')
plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Loss", linestyle='--', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig('./loss.png') # Required for your 'fig' folder
plt.show()

# ===== Plotting Training and Validation IoU =====
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_iou_history, label="Training IoU", color='green')
plt.plot(range(1, epochs + 1), val_iou_history, label="Validation IoU", linestyle='--', color='red')
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Training and Validation IoU over Epochs")
plt.legend()
plt.grid(True)
plt.savefig('./iou.png') # Required for your 'fig' folder
plt.show()