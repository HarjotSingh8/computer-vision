import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import random

def run_segmentation(image_path, sam_checkpoint="sam_vit_h.pth", model_type="vit_h", device="cuda"):
    """
    Runs segmentation on an image and returns the generated masks.

    Args:
        image_path (str): Path to the input image.
        sam_checkpoint (str): Path to the SAM model checkpoint.
        model_type (str): Type of SAM model to use.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list: List of generated masks.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks
    masks = mask_generator.generate(image)

    return masks


def test_segmentation():
    # Pick a random image from the directory
    image_dir = "./data/Small/"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No images found in the directory.")
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)

    # Run segmentation
    masks = run_segmentation(image_path)

    # Save the output
    output_path = f"./output/run_{random.randint(1, 1000)}.png"
    plt.imshow(masks[0]['segmentation'], cmap='gray')  # Assuming masks[0] exists and has 'segmentation'
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Segmentation output saved to {output_path}")

if __name__ == "__main__":
    # Test the segmentation function
    test_segmentation()