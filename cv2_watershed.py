import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def run_segmentation_watershed(image_path):
    """
    Runs segmentation on an image using OpenCV's watershed algorithm and returns the segmented image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Segmented image with all segments visualized and bounding boxes drawn.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise and separate objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Determine sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Determine sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Determine unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    segmented_image = image.copy()

    # Assign random colors to each segment and draw bounding boxes
    unique_markers = np.unique(markers)
    # First pass: Assign random colors to each segment
    for marker in unique_markers:
        if marker == -1:  # Skip boundary markers
            continue
        color = [random.randint(0, 255) for _ in range(3)]
        segmented_image[markers == marker] = color

    # Second pass: Draw bounding boxes for each segment
    for marker in unique_markers:
        if marker == -1:  # Skip boundary markers
            continue
        mask = (markers == marker).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White bounding box

    return segmented_image

def test_segmentation_watershed():
    # Pick a random image from the directory
    image_dir = "./data/Small/"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("No images found in the directory.")
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)

    # Run segmentation
    segmented_image = run_segmentation_watershed(image_path)

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

def perf_test_watershed():
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
        _ = run_segmentation_watershed(image_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_images / elapsed_time
    print(f"Processed {num_images} images in {elapsed_time:.2f} seconds. FPS: {fps:.2f}")

if __name__ == "__main__":
    # Test the segmentation function
    test_segmentation_watershed()
    # Run a performance test
    # perf_test_watershed()