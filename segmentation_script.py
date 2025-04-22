import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import random

sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"

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
    masks = run_segmentation(image_path, sam_checkpoint=sam_checkpoint, model_type=model_type, device="cuda")

    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Save the original image
    original_output_path = f"./output/output_{random_image}.original.png"
    plt.imshow(original_image)
    plt.axis('off')
    plt.savefig(original_output_path)
    print(f"Original image saved to {original_output_path}")

    # Overlay all masks on the image
    overlay = original_image.copy()
    for mask in masks:
        segmentation = mask['segmentation']
        color = [random.randint(0, 255) for _ in range(3)]  # Random color for each mask
        overlay[segmentation] = color

    # Save the output
    overlay_output_path = f"./output/output_{random_image}.png"
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(overlay_output_path)
    print(f"Overlay output saved to {overlay_output_path}")

def perf_test():
    # test fps
    num_images = 10
    image_dir = "./data/Small/"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No images found in the directory.")
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Load the SAM model
    # sam_checkpoint = "sam_vit_h.pth"
    # model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)
    # Generate masks
    import time
    start_time = time.time()
    for _ in range(num_images):
        print(f"Processing image {_ + 1}/{num_images}")
        masks = mask_generator.generate(image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_images / elapsed_time
    print(f"Processed {num_images} images in {elapsed_time:.2f} seconds. FPS: {fps:.2f}")

if __name__ == "__main__":
    # Test the segmentation function
    test_segmentation()
    # run a performance test
    perf_test()