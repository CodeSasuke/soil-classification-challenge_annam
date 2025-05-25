"""
Author: Siddhant Bhardwaj
Team Name: Siddhant Bhardwaj
Team Members: Siddhant Bhardwaj, Sivadhanushya
Leaderboard Rank: 36
"""

# This file outlines preprocessing details for the one-class soil classification task.
# The primary image preprocessing is handled by the CLIPProcessor from the
# Hugging Face Transformers library, which is specific to the pre-trained CLIP model.

from PIL import Image
from transformers import CLIPProcessor
import torch

# --- Configuration (should match your main script's model choice) ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP Processor ---
try:
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print(f"CLIP Processor for '{CLIP_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP Processor: {e}")
    clip_processor = None

def preprocess_single_image_for_clip(image_path):
    if not clip_processor:
        print("CLIP Processor not loaded. Cannot preprocess image.")
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True)
        return inputs.pixel_values
        
    except FileNotFoundError:
        print(f"Preprocessing Error: Image not found at '{image_path}'.")
        return None
    except Exception as e:
        print(f"Preprocessing Error for image '{image_path}': {e}.")
        return None

def main_preprocessing_workflow_summary():
    print("\n--- Preprocessing Workflow Summary ---")
    print("1. Image Loading: Images are loaded using PIL (Pillow) and converted to RGB format.")
    print("2. CLIP-Specific Transformation: The `CLIPProcessor` handles:")
    print("   - Resizing images to the expected input dimensions of the CLIP image encoder")
    print("   - Normalizing pixel values according to the pre-training statistics")
    print("   - Converting the image into a PyTorch tensor format.")
    print("3. No further dataset-wide preprocessing steps are applied")
    print("------------------------------------")
    return 0

if __name__ == '__main__':
    print("--- Running Preprocessing File ---")
    main_preprocessing_workflow_summary()
    print("--- Preprocessing File Finished ---")
