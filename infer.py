import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model import GeneratorUNet  # Import the Generator model

# Define device (Use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model safely
generator = GeneratorUNet()
checkpoint = torch.load("checkpoints/generator_epoch_50.pth", map_location=device, weights_only=True)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.to(device)
generator.eval()

# Define preprocessing and postprocessing transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

reverse_transform = transforms.Compose([
    transforms.ToPILImage(),
])

def find_images_in_subfolders(root_folder):
    """Recursively find all image files in subdirectories"""
    image_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'JPEG')):
                image_files.append(os.path.join(root, file))
    return image_files

def verify_image(file_path):
    """Checks if the file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception as e:
        print(f"❌ Skipping invalid image: {file_path} ({e})")
        return False

def colorize_image(input_path, output_path):
    """Loads a colored image, converts it to grayscale, colorizes it using the trained model, and saves the output."""
    
    # Verify the image before processing
    if not verify_image(input_path):
        return
    
    try:
        image = Image.open(input_path).convert("RGB")  # Ensure image is RGB
        grayscale = image.convert("L")  # Convert to grayscale
    except Exception as e:
        print(f"❌ Error loading image {input_path}: {e}")
        return
    
    grayscale = transform(grayscale).unsqueeze(0).to(device)  # Apply grayscale transformation

    # Generate colorized image
    with torch.no_grad():
        colorized = generator(grayscale)
        colorized_image = reverse_transform(colorized.squeeze(0).cpu())

    # Save output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colorized_image.save(output_path)
    print(f"✅ Colorized image saved at: {output_path}")

def process_images(input_path, output_folder):
    """Processes either a single image or all images in a directory."""
    if os.path.isdir(input_path):  # If input is a folder, process all images
        print(f"📂 Processing all images in directory: {input_path}")
        
        os.makedirs(output_folder, exist_ok=True)
        images = find_images_in_subfolders(input_path)  # Search for images in subfolders
        
        if len(images) == 0:
            print("❌ No valid image files found in the directory!")
            return
        
        print(f"📸 Found {len(images)} images. Processing...")
        for idx, filename in enumerate(images, start=1):
            output_file = os.path.join(output_folder, f"colorized_{os.path.basename(filename)}")
            print(f"🔹 [{idx}/{len(images)}] Processing: {filename}")
            colorize_image(filename, output_file)
    
    elif os.path.isfile(input_path):  # If input is a single file
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "colorized.png")
        colorize_image(input_path, output_file)
    
    else:
        print(f"❌ Error: '{input_path}' does not exist!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize a grayscale image using the trained GAN model.")
    parser.add_argument("--input", type=str, default=r"E:/AI Image Colorizer/datasets/imagenetmini-1000/val",
                        help="Path to a single grayscale image or folder containing multiple images")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Folder to save the colorized images")

    args = parser.parse_args()

    # Run image processing
    process_images(args.input, args.output)
