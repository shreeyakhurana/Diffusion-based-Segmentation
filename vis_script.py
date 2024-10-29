import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

def tensor_to_image(tensor, output_path=None):
    # If batch dimension exists, take the first image
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    img_np = tensor.numpy().transpose(1, 2, 0)
    
    if img_np.shape[2] == 1:  # Grayscale
        img_np = np.squeeze(img_np)
    elif img_np.shape[2] == 3:  # RGB
        pass
    else:
        raise ValueError(f"Unsupported number of channels: {img_np.shape[2]}")
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    if output_path:
        img_pil.save(output_path)
    
    return img_pil

# Load result image tensors from results folder
for f in os.listdir('results'):
    result_image = torch.load(f'results/{f}', map_location=torch.device('cpu'))
    tensor_to_image(result_image, f'results_images/{f}.jpg')
