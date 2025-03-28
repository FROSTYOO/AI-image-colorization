# AI Image Colorizer

## Overview
This project implements an AI-based image colorizer using Generative Adversarial Networks (GANs). The model takes grayscale images as input and predicts the corresponding colorized versions.

## Features
- **Deep Learning Model**: Uses a UNet-based Generator and a Discriminator to improve colorization quality.
- **Custom Dataset Handling**: Supports grayscale image datasets for training.
- **GAN-based Approach**: Trained using adversarial loss and L1 loss for realistic colorization.
- **Training Pipeline**: Includes automatic checkpoint saving and sample image generation.
- **Inference Script**: Allows colorization of new grayscale images.
- **Web Interface**: A FastAPI-based web app for real-time image colorization.

## Installation
### Prerequisites
Ensure you have Python 3.7+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Additional Dependencies
If not using `requirements.txt`, install dependencies manually:
```bash
pip install torch torchvision fastapi uvicorn tqdm pillow numpy matplotlib```

## Usage
### Training the Model
To train the AI Image Colorizer, run:
```bash
python train.py
```
This will train the model using the dataset defined in `train.py` and save checkpoints in the `checkpoints/` directory.

### Running Inference
To colorize grayscale images using the trained model, run:
```bash
python infer.py --input <grayscale_image_path> --output <colorized_image_path>
```

### Running the Web App
A FastAPI-based web interface allows real-time image colorization:
```bash
uvicorn fastapp:app --host 0.0.0.0 --port 8000
```
Then, open `http://localhost:8000/docs` to test the API.

## Project Structure
```
.
├── dataset.py          # Custom dataset loader
├── fastapp.py          # Web app using FastAPI
├── infer.py            # Inference script for image colorization
├── model.py            # Generator and Discriminator models
├── train.py            # Training script for AI model
├── checkpoints/        # Directory for saving trained models
└── README.md           # Project documentation
```

## Model Details
- **Generator**: A UNet-based architecture to generate colorized images.
- **Discriminator**: A CNN-based model to distinguish real vs. fake colorized images.
- **Loss Functions**: Uses Binary Cross-Entropy (BCE) loss for GAN training and L1 loss for pixel-wise accuracy.

## Future Improvements
- Enhance model performance with larger datasets.
- Improve inference speed for real-time applications.
- Deploy as a web service using Flask/Streamlit.

## License
This project is open-source under the MIT License.

## Author
[Sauman Sarkar](https://github.com/FROSTYOO)

