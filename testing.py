import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Setup
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple UNet
class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleUNet, self).__init__()
        # Encoder: Pretrained ResNet18 [cite: 608]
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # Decoder [cite: 612]
        self.decoder = nn.Sequential(
        # Stage 1: 8x8 -> 16x16
        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        # Stage 2: 16x16 -> 32x32
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        # Stage 3: 32x32 -> 64x64
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        # Stage 4: 64x64 -> 128x128 (Added)
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        # Stage 5: 128x128 -> 256x256 (Added)
        nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        # Final layer to match num_classes
        nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Resize back to original UAV image size (600, 800) [cite: 635]
        return nn.functional.interpolate(x, size=(600, 800), mode="bilinear")
    
# Path settings
test_img_dir = "./UAV/dataset/test/imgs" # Path to your 1000 test images
output_dir = "./UAV/dataset/test/masks"     # Where to save the created masks
os.makedirs(output_dir, exist_ok=True)

# 1. Load Model
model = SimpleUNet(num_classes=16).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 2. Run Inference
with torch.no_grad():
    for img_name in os.listdir(test_img_dir):
        # Load and preprocess
        img_path = os.path.join(test_img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use the SAME transform you used in training (Resize to 256x256)
        input_tensor = transform(image).unsqueeze(0).to(device) 
        
        # Predict
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 3. Save as grayscale image
        # Mask values are 0-15, which are very dark. You may want to save them as raw values.
        mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        mask_img.save(os.path.join(output_dir, img_name))