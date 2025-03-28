import sys
import os
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import uvicorn
import uuid  # For generating unique filenames

# ✅ Add `src/` directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# ✅ Import trained model
from model import GeneratorUNet  

# Initialize FastAPI app
app = FastAPI(title="AI Image Colorizer API", description="Upload a grayscale image to get a colorized output!")

# ✅ Serve the "static/" folder to make images accessible in the browser
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

reverse_transform = transforms.ToPILImage()

# Store last uploaded images for deletion
last_uploaded = None
last_colorized = None

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serves an HTML page with drag-and-drop file upload and remove button."""
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Image Colorizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script>
            async function uploadImage(event) {{
                let formData = new FormData();
                formData.append("file", event.target.files[0]);

                const response = await fetch("/colorize/", {{ method: "POST", body: formData }});
                const result = await response.json();

                if (response.ok) {{
                    document.getElementById("inputImage").src = result.input_image_url;
                    document.getElementById("colorizedImage").src = result.colorized_image_url;
                    document.getElementById("downloadBtn").href = result.colorized_image_url;
                    document.getElementById("imageContainer").style.display = "block";
                }} else {{
                    alert("Error: " + result.message);
                }}
            }}

            async function removeImages() {{
                const response = await fetch("/remove/", {{ method: "DELETE" }});

                if (response.ok) {{
                    document.getElementById("imageContainer").style.display = "none";
                    document.getElementById("fileInput").value = "";  // Clear input
                }}
            }}

            function allowDragOver(event) {{
                event.preventDefault();
                document.getElementById("drop-area").classList.add("drag-over");
            }}

            function removeDragOver(event) {{
                event.preventDefault();
                document.getElementById("drop-area").classList.remove("drag-over");
            }}

            function handleDrop(event) {{
                event.preventDefault();
                document.getElementById("drop-area").classList.remove("drag-over");
                let fileInput = document.getElementById("fileInput");
                fileInput.files = event.dataTransfer.files;
                uploadImage({{ target: fileInput }});
            }}
        </script>
        <style>
            body {{
                background-color: #f8f9fa;
                text-align: center;
                font-family: Arial, sans-serif;
            }}
            .container {{
                max-width: 500px;
                margin-top: 50px;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }}
            .btn-upload {{ width: 100%; }}
            .footer {{ margin-top: 20px; font-size: 14px; color: gray; }}
            #imageContainer {{ margin-top: 20px; display: none; }}
            img {{ max-width: 100%; border-radius: 10px; box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1); }}
            #drop-area {{
                border: 2px dashed #007bff;
                padding: 20px;
                border-radius: 10px;
                cursor: pointer;
                background-color: #e9f5ff;
            }}
            #drop-area.drag-over {{ background-color: #cce5ff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎨 AI Image Colorizer</h2>
            <div id="drop-area" ondragover="allowDragOver(event)" ondragleave="removeDragOver(event)" ondrop="handleDrop(event)">
                <p>📂 Drag & Drop Image Here</p>
                <input id="fileInput" type="file" name="file" accept="image/*" class="form-control mb-3" onchange="uploadImage(event)">
                <button class="btn btn-primary btn-upload">Colorize Image</button>
            </div>

            <div id="imageContainer">
                <h5>📷 Uploaded & Colorized Image</h5>
                <img id="inputImage" src="" alt="Uploaded Image">
                <img id="colorizedImage" src="" alt="Colorized Image">
                <a id="downloadBtn" href="#" class="btn btn-success">📥 Download</a>
                <button class="btn btn-danger" onclick="removeImages()">🗑 Remove</button>
            </div>
        </div>
    </body>
    </html>
    """)

from PIL import ImageEnhance, ImageFilter

@app.post("/colorize/")
async def colorize_image(file: UploadFile = File(...)):
    """Upload a grayscale image and receive an enhanced colorized version."""
    global last_uploaded, last_colorized

    # Remove old images
    if last_uploaded and os.path.exists(last_uploaded):
        os.remove(last_uploaded)
    if last_colorized and os.path.exists(last_colorized):
        os.remove(last_colorized)

    # ✅ Open image and convert to grayscale
    input_image = Image.open(BytesIO(await file.read())).convert("L")

    # ✅ Preprocess image (normalize to [-1, 1])
    grayscale = transform(input_image).unsqueeze(0).to(device)

    # ✅ Generate colorized image
    with torch.no_grad():
        colorized = generator(grayscale)
        colorized = (colorized * 0.5) + 0.5  # Scale from [-1,1] to [0,1]
        colorized = colorized.clamp(0, 1)  # Ensure values are valid

    # ✅ Convert tensor to PIL image
    colorized_image = reverse_transform(colorized.squeeze(0).cpu()).convert("RGB")

    # ✅ Apply Image Enhancements
  # ✅ Apply Improved Enhancements
    enhancer = ImageEnhance.Sharpness(colorized_image)
    colorized_image = enhancer.enhance(2.0)  # More sharpness

    enhancer = ImageEnhance.Sharpness(colorized_image)
    colorized_image = enhancer.enhance(2.5)  # Stronger sharpening
   # Stronger contrast

    enhancer = ImageEnhance.Color(colorized_image)
    colorized_image = enhancer.enhance(1.3)  # Reduce color boost
        # Boost color richness

    enhancer = ImageEnhance.Brightness(colorized_image)
    colorized_image = enhancer.enhance(0.9)  # Reduce brightness slightly
                                             #  Slightly brighter


    # ✅ Save input and output images
    input_filename = f"static/input_{uuid.uuid4().hex[:8]}.png"
    output_filename = f"static/colorized_{uuid.uuid4().hex[:8]}.png"

    os.makedirs("static", exist_ok=True)
    input_image.save(input_filename)
    colorized_image.save(output_filename)

    # Store last images for deletion later
    last_uploaded, last_colorized = input_filename, output_filename

    return JSONResponse(content={
        "message": "Success",
        "input_image_url": f"/{input_filename}",
        "colorized_image_url": f"/{output_filename}"
    })

@app.delete("/remove/")
async def remove_images():
    """Delete the last uploaded and colorized images."""
    global last_uploaded, last_colorized

    if last_uploaded and os.path.exists(last_uploaded):
        os.remove(last_uploaded)
        last_uploaded = None

    if last_colorized and os.path.exists(last_colorized):
        os.remove(last_colorized)
        last_colorized = None

    return JSONResponse(content={"message": "Images removed successfully"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
