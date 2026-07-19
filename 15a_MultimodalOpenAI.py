"""
PIL is the Python Imaging Library
adds image processing capabilities to your Python interpreter.
Provides extensive file format support


CLIPModel and CLIPProcessor are key components in the Hugging Face transformers library used to run 
the CLIP (Contrastive Language-Image Pre-training) AI model. They help bridge the gap between images 
and text so computers can understand them together"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from dotenv import load_dotenv

from huggingface_hub import utils
utils.disable_progress_bars()

load_dotenv()

# Setup Hugging Face Key Safely
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# 1. Load the model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
# Changes the behavior so model acts in deployment mode rather than training mode.
model.eval() 

# 2. Define the list of all four local images
image_paths = [
    "data/iwood_cabin.jpg",
    "data/icar-highway.jpg",
    "data/igoa-beach.jpg",
    "data/icouple-dancing.jpg"
]

# 3. Define the text descriptions to test against every image
text_queries = [
    "a cozy wooden cabin surrounded by green trees", 
    "a modern sports car driving on a highway",
    "a couple dancing on street",
    "beach in goa india"
]

print("Starting multimodal evaluation loop...\n")

# Loop through each image path one by one
for image_path in image_paths:
    # Safety check to make sure the file exists before attempting to open it
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Could not find {image_path}. Skipping this image.")
        continue

    print(f"--------------------------------------------------")
    # Open the image locally and convert it to standard RGB
    # `convert("RGB")`` is used because the model strictly expects exactly 3 color channels (Red, Green, Blue)
    #   -Handles transparency (RGBA) and grayscale formats safely

    local_image = Image.open(image_path).convert("RGB")
    # print(local_image.format, local_image.size, local_image.mode) # NOTE Just for testing

    # 4. Format the current image and all text queries into tensors
    inputs = processor(
        text=text_queries, 
        images=local_image, 
        return_tensors="pt", 
        padding=True
    )

    # 5. Send inputs to the model
    # `torch.no_grad()` - Disables the mechanism that tracks mathematical gradients. Saves memory and speeds up execution.
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. Turn the raw scores into match percentages
    # `logits_per_image` as raw, unformatted "confidence points" that the model assigns to each text description. 
    logits_per_image = outputs.logits_per_image  
    # print("logits_per_image", logits_per_image) # NOTE Just for testing
    # Softmax is a mathematical tool that turns those raw points into clean, readable percentages that all add up to 100%.
    # dim=-1: calculate these percentages across the rows
    probs = logits_per_image.softmax(dim=-1)  
    # probs as a two-dimensional grid. 
    # print("probs", probs[0]) # NOTE Just for testing
    

    # 7. Print the final verdict for this specific image
    print(f"--- CLIP's Verdict for {os.path.basename(image_path)} ---")
    # `zip()` acts like a physical zipper, pairing up elements from two lists side-by-side so
    # ex. ("a cozy wooden cabin...", 99.2%)
    for text, probability in zip(text_queries, probs[0]):
        if probability.item()>0.8:
            print(f"Match probability for '{text}': {probability.item() * 100:.2f}%")
    print()



