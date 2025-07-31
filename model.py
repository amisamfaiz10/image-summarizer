# model.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Move model to device
model.to(device)
