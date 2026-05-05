#!/usr/bin/env python3
"""
Image Dataset Resizer

Resizes all images in a dataset folder while preserving the folder structure.
Each class should be a subfolder in the input directory.

Features:
- Resizes images to a fixed size (default: 224x224)
- Preserves aspect ratio with padding
- Supports .jpg, .jpeg, .png, .webp
- Skips and logs corrupted/unreadable files
- Displays a progress bar
- Prints a summary (total processed, skipped, per-class count)

Dependencies:
- Pillow
- tqdm

Install:
    pip install pillow tqdm

Usage:
    python resize_dataset.py input_dataset output_dataset

Example:
    python resize_dataset.py data/train data_resized/train
"""

import os
import sys
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import defaultdict

# ==============================
# CONFIGURATION
# ==============================
TARGET_SIZE = (224, 224)  # (width, height)
PADDING_COLOR = (0, 0, 0)  # Black padding (use (255,255,255) for white)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# ==============================
# CORE FUNCTIONS
# ==============================
def resize_with_padding(img, target_size, padding_color):
    """
    Resize an image while preserving aspect ratio and add padding.

    Args:
        img (PIL.Image): Input image
        target_size (tuple): (width, height)
        padding_color (tuple): RGB color for padding

    Returns:
        PIL.Image: Resized and padded image
    """
    target_w, target_h = target_size
    original_w, original_h = img.size

    # Compute scaling factor
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize image
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create new image with padding
    new_img = Image.new("RGB", (target_w, target_h), padding_color)

    # Center the image
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def process_dataset(input_dir, output_dir):
    """
    Process the dataset and resize all images.

    Args:
        input_dir (str): Path to input dataset
        output_dir (str): Path to output dataset
    """
    total_processed = 0
    total_skipped = 0
    per_class_count = defaultdict(int)

    # Gather all image paths
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                image_paths.append(os.path.join(root, file))

    # Progress bar
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Open image
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                # Resize with padding
                img_resized = resize_with_padding(
                    img, TARGET_SIZE, PADDING_COLOR
                )

            # Compute output path
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save image
            img_resized.save(output_path)

            # Update counters
            total_processed += 1
            class_name = rel_path.split(os.sep)[0]
            per_class_count[class_name] += 1

        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[WARNING] Skipping file: {img_path} ({e})")
            total_skipped += 1

    # ==============================
    # SUMMARY
    # ==============================
    print("\n===== SUMMARY =====")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")
    print("\nPer-class counts:")
    for cls, count in per_class_count.items():
        print(f"  {cls}: {count}")


# ==============================
# ENTRY POINT
# ==============================
def main():
    if len(sys.argv) != 3:
        print("Usage: python resize_dataset.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    process_dataset(input_dir, output_dir)


if __name__ == "__main__":
    main()