import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from tqdm import tqdm
from model import GeneratorUNet, Discriminator  # Import models
from dataset import GrayscaleColorizationDataset  # Import dataset

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset path
data_dir = r"E:\AI Image Colorizer\datasets\imagenetmini-1000\train"

# Define dataset transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to train the model
def train(num_epochs=50):
    # Load dataset
    dataset = GrayscaleColorizationDataset(data_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize models
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Function to save model checkpoints
    def save_checkpoint(model, optimizer, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

    # Training Loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        loop = tqdm(train_loader, leave=True)
        for grayscale, real_color in loop:
            grayscale, real_color = grayscale.to(device), real_color.to(device)

            # Train Discriminator
            fake_color = generator(grayscale)
            real_pred = discriminator(grayscale, real_color)
            fake_pred = discriminator(grayscale, fake_color.detach())

            d_loss_real = criterion_gan(real_pred, torch.ones_like(real_pred))
            d_loss_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            fake_pred = discriminator(grayscale, fake_color)
            g_gan_loss = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_l1_loss = criterion_l1(fake_color, real_color) * 100
            g_loss = g_gan_loss + g_l1_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(D_Loss=d_loss.item(), G_Loss=g_loss.item())

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(generator, g_optimizer, epoch, f"generator_epoch_{epoch+1}.pth")
            save_checkpoint(discriminator, d_optimizer, epoch, f"discriminator_epoch_{epoch+1}.pth")

            # Save sample output images
            save_image(fake_color[:8], f"checkpoints/sample_epoch_{epoch+1}.png", nrow=4, normalize=True)

# ✅ Ensure script runs correctly on Windows
if __name__ == "__main__":
    train(num_epochs=50)
