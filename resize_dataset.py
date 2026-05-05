#!/usr/bin/env python3
"""
Dataset Image Processor: Resize + Rename

This script processes a dataset structured as:
    root/
        class1/
            img1.jpg
            img2.png
        class2/
            img3.jpg
            ...

Features:
- Resizes images to a fixed size (default: 224x224) with padding
- Renames images to: {class_name}_{index:04d}.ext
- Index restarts at 1 for each class
- Preserves original extension
- Skips files if target name already exists
- Dry-run mode for renaming
- Skips corrupted/unreadable images
- Shows progress bar
- Prints summary (total processed, renamed, skipped, per-class)

Dependencies:
    pip install pillow tqdm

Usage:
    python resize_dataset.py data/input_dataset data/output_dataset
    python resize_dataset.py data/input_dataset data/output_dataset --dry-run


"""

import os
import sys
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import defaultdict

# ==============================
# CONFIGURATION
# ==============================
TARGET_SIZE = (224, 224)
PADDING_COLOR = (0, 0, 0)  # (255,255,255) for white
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# ==============================
# IMAGE PROCESSING
# ==============================
def resize_with_padding(img, target_size, padding_color):
    """Resize while keeping aspect ratio + padding."""
    target_w, target_h = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (target_w, target_h), padding_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


# ==============================
# MAIN PROCESS
# ==============================
def process_dataset(input_dir, output_dir, dry_run=False):
    total_processed = 0
    total_renamed = 0
    total_skipped = 0

    per_class_count = defaultdict(int)

    # Iterate over class folders
    for class_name in sorted(os.listdir(input_dir)):
        class_input_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # Collect images in this class
        images = [
            f for f in sorted(os.listdir(class_input_path))
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

        index = 1

        for filename in tqdm(images, desc=f"{class_name}", leave=False):
            input_path = os.path.join(class_input_path, filename)

            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{class_name}_{index:04d}{ext}"
            output_path = os.path.join(class_output_path, new_name)

            # Check overwrite
            if os.path.exists(output_path):
                print(f"[SKIP] Exists: {output_path}")
                total_skipped += 1
                index += 1
                continue

            if dry_run:
                print(f"[DRY-RUN] {input_path} -> {output_path}")
                total_renamed += 1
                per_class_count[class_name] += 1
                index += 1
                continue

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img_resized = resize_with_padding(
                        img, TARGET_SIZE, PADDING_COLOR
                    )

                img_resized.save(output_path)

                total_processed += 1
                total_renamed += 1
                per_class_count[class_name] += 1

            except (UnidentifiedImageError, OSError, ValueError) as e:
                print(f"[WARNING] Skipping: {input_path} ({e})")
                total_skipped += 1

            index += 1

    # ==============================
    # SUMMARY
    # ==============================
    print("\n===== SUMMARY =====")
    print(f"Total processed: {total_processed}")
    print(f"Total renamed:   {total_renamed}")
    print(f"Total skipped:   {total_skipped}")

    print("\nPer-class counts:")
    for cls, count in per_class_count.items():
        print(f"  {cls}: {count}")


# ==============================
# ENTRY POINT
# ==============================
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_dir> <output_dir> [--dry-run]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dry_run = "--dry-run" in sys.argv

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    process_dataset(input_dir, output_dir, dry_run)


if __name__ == "__main__":
    main()