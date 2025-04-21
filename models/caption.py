from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model and processor once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

@torch.no_grad()
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

