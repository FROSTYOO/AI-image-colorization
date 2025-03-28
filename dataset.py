import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import os

# Define dataset transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor (range [0,1])
])

class GrayscaleColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Load image
        grayscale = transforms.Grayscale(num_output_channels=1)(img)  # Convert to grayscale
        return grayscale, img  # Return (grayscale input, color target)

# Define dataset path
data_dir = r"E:\AI Image Colorizer\datasets\imagenetmini-1000"

# Load dataset
dataset = GrayscaleColorizationDataset(data_dir, transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Use CUDA acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
