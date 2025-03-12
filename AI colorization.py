import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use mixed precision training for better performance
from torch.cuda.amp import autocast, GradScaler

# Custom Dataset Class with prefetching
class ColorizationDataset(Dataset):
    def __init__(self, image_paths, transform=None, img_size=(224, 224)):
        self.image_paths = image_paths
        self.transform = transform
        self.img_size = img_size
        
        # Prefetch and cache data if dataset isn't too large
        self.cache = {}
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
            
        # Load and process image
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            # Handle corrupted images
            print(f"Warning: Could not read image {self.image_paths[idx]}")
            # Use an empty image of the right size as a fallback
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Resize image first to save computation
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
        
        L = img[:, :, 0:1]  # Extract L channel (1-channel)
        AB = img[:, :, 1:]  # Extract AB channels (2-channel)
        
        # Convert to float and normalize
        L = L.astype(np.float32) / 255.0
        AB = AB.astype(np.float32) / 128.0  # Normalize to [-1, 1]
        
        # Convert to tensor
        L = torch.tensor(L).permute(2, 0, 1)  # Change shape to (1, H, W)
        AB = torch.tensor(AB).permute(2, 0, 1)  # Change shape to (2, H, W)
        
        if self.transform:
            L = self.transform(L)
            AB = self.transform(AB)
        
        # Cache for future retrieval
        self.cache[idx] = (L, AB)
        
        return L, AB

# Optimized Colorization Model with skip connections
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        
        # Encoder (Feature Extraction)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify input layer
        
        # Use layers as separate modules to extract intermediate features for skip connections
        self.encoder_stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder_stage2 = resnet.layer1
        self.encoder_stage3 = resnet.layer2
        self.encoder_stage4 = resnet.layer3
        self.encoder_stage5 = resnet.layer4
        
        # Decoder (Upsampling with skip connections)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Added an extra upsampling layer to match input size (224x224)
        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder_stage1(x)  # 56x56
        e2 = self.encoder_stage2(e1)  # 56x56
        e3 = self.encoder_stage3(e2)  # 28x28
        e4 = self.encoder_stage4(e3)  # 14x14
        e5 = self.encoder_stage5(e4)  # 7x7
        
        # Decoder with skip connections
        d1 = self.upsample1(e5)  # 14x14
        d1 = torch.cat([d1, e4], dim=1)
        
        d2 = self.upsample2(d1)  # 28x28
        d2 = torch.cat([d2, e3], dim=1)
        
        d3 = self.upsample3(d2)  # 56x56
        d3 = torch.cat([d3, e2], dim=1)
        
        d4 = self.upsample4(d3)  # 112x112
        
        # Additional upsampling to reach 224x224
        d5 = self.upsample5(d4)  # 224x224
        
        output = self.output_layer(d5)  # 224x224, 2 channels
        
        return output

# Model save and load function
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Define Hyperparameters
batch_size = 32  # Increased batch size
epochs = 20
learning_rate = 0.001

# Training function
def train_model(image_folder, output_model_path, img_size=(224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Preprocessing - simplified transform
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))  # Normalize L channel
    ])
    
    # Image Folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Load Dataset
    dataset = ColorizationDataset(image_paths, transform, img_size=img_size)
    
    # Use num_workers for faster data loading
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True)
    
    # Initialize Model, Loss, Optimizer
    model = ColorizationNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training Loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for L, AB in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            L, AB = L.to(device), AB.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision for forward pass
            with autocast():
                output = model(L)
                # No need for resize now - model outputs 224x224
                loss = criterion(output, AB)
            
            # Use scaler for backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, output_model_path)
            print(f"Model saved to {output_model_path}")
    
    return model

# Function to colorize a single image
def colorize_image(model, image_path, device, output_size=None):
    model.eval()  # Set model to evaluation mode
    
    # Load & Convert Image to LAB Color Space
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    original_size = (img.shape[1], img.shape[0])  # Save original size
    
    # Resize for model input
    img_resized = cv2.resize(img, (224, 224))
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB).astype("float32")
    
    # Extract L Channel and Normalize to [0, 1]
    L = img_lab[:, :, 0] / 255.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: [1, 1, H, W]

    # Predict AB Channels
    with torch.no_grad():
        AB_pred = model(L_tensor)  # Model outputs [-1, 1] range
        
        # Resize to output size if specified, otherwise original size
        if output_size:
            target_size = output_size
        else:
            target_size = original_size
            
        AB_pred = torch.nn.functional.interpolate(
            AB_pred, size=target_size, mode="bilinear", align_corners=False)
        AB_pred = AB_pred.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert to NumPy and transpose to (H, W, 2)

    # Convert to original image size
    orig_img = cv2.resize(img, target_size)
    orig_img_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB)
    
    # Convert AB Channels to LAB Range (-128, 127)
    AB_pred = (AB_pred * 128).astype(np.int16)  # Scale model output to LAB format

    # Combine L from original and Predicted AB Channels
    img_pred_lab = np.zeros_like(orig_img_lab)
    img_pred_lab[:, :, 0] = orig_img_lab[:, :, 0]  # Use original L channel for better details
    img_pred_lab[:, :, 1:] = np.clip(AB_pred, -128, 127)  # Ensure AB values are in LAB range

    # Convert LAB to BGR
    img_pred_bgr = cv2.cvtColor(img_pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return img_pred_bgr

# Main execution for training
if __name__ == "__main__":
    image_folder = r"E:\\AI Image Colorization\\path_to_images"
    output_model_path = "colorization_model.pth"
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Training with GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, training with CPU")
    
    # Train the model
    train_model(image_folder, output_model_path)
    
    # Example Usage for colorizing an image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationNet().to(device)
    model = load_model(model, output_model_path, device)
    
    test_image = "test_image.jpg"
    colorized_img = colorize_image(model, test_image, device)
    
    # Display the Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB))
    plt.title("Colorized")
    plt.axis("off")
    
    plt.savefig("colorization_result.jpg")
    plt.show()