import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt



def run_segmentation_cv2(image_path, threshold_method=cv2.THRESH_BINARY):
    """
    Runs segmentation on an image using OpenCV and returns the segmented image.

    Args:
        image_path (str): Path to the input image.
        threshold_method (int): OpenCV thresholding method to use.

    Returns:
        np.ndarray: Segmented image.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the specified thresholding method
    _, binary = cv2.threshold(blurred, 127, 255, threshold_method)  # Adjusted threshold value to 127

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the segmented regions
    mask = np.zeros_like(image)
    for contour in contours:
        color = [random.randint(0, 255) for _ in range(3)]  # Random color for each contour
        cv2.drawContours(mask, [contour], -1, color, thickness=cv2.FILLED)

    return mask

def test_segmentation_cv2(threshold_method=cv2.THRESH_BINARY):
    # Pick a random image from the directory
    image_dir = "./data/Small/"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No images found in the directory.")
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)

    # Run segmentation
    segmented_image = run_segmentation_cv2(image_path, threshold_method=threshold_method)

    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Save the original image
    original_output_path = f"./output/output_{random_image}.original.png"
    plt.imshow(original_image)
    plt.axis('off')
    plt.savefig(original_output_path)
    print(f"Original image saved to {original_output_path}")

    # Save the segmented image
    segmented_output_path = f"./output/output_{random_image}.png"
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.imshow(segmented_image_rgb)
    plt.axis('off')
    plt.savefig(segmented_output_path)
    print(f"Segmented output saved to {segmented_output_path}")

def perf_test_cv2(threshold_method=cv2.THRESH_BINARY):
    # Test FPS
    num_images = 10
    image_dir = "./data/Small/"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No images found in the directory.")
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)

    # Load the image
    image = cv2.imread(image_path)

    # Measure performance
    import time
    start_time = time.time()
    for _ in range(num_images):
        print(f"Processing image {_ + 1}/{num_images}")
        _ = run_segmentation_cv2(image_path, threshold_method=threshold_method)
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_images / elapsed_time
    print(f"Processed {num_images} images in {elapsed_time:.2f} seconds. FPS: {fps:.2f}")

if __name__ == "__main__":
    # Test the segmentation function
    test_segmentation_cv2()
    # Run a performance test
    # perf_test_cv2()