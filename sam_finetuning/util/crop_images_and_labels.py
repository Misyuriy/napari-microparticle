import os
import sys
import argparse
from PIL import Image


TARGET_WIDTH = 1280
TARGET_HEIGHT = 960
CROP_BOTTOM = 64


def crop_bottom(image_path, output_path, crop_height):
    with Image.open(image_path) as img:
        width, height = img.size

        if width != TARGET_WIDTH or height != TARGET_HEIGHT + CROP_BOTTOM:
            print(f"Warning: {image_path} has size {width}x{height}, expected {TARGET_WIDTH}x{TARGET_HEIGHT + CROP_BOTTOM}")

        cropped = img.crop((0, 0, width, height - crop_height))
        cropped.save(output_path)
        print(f"Saved cropped image: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop bottom 64 pixels from images and masks."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing 'images' and 'labels' subfolders."
    )
    args = parser.parse_args()

    root_dir = args.root_dir

    input_images_dir = os.path.join(root_dir, "images")
    input_labels_dir = os.path.join(root_dir, "labels")
    output_images_dir = os.path.join(root_dir, "images_cropped")
    output_labels_dir = os.path.join(root_dir, "labels_cropped")

    if not os.path.isdir(input_images_dir):
        print(f"Error: Images directory not found: {input_images_dir}")
        sys.exit(1)
    if not os.path.isdir(input_labels_dir):
        print(f"Error: Labels directory not found: {input_labels_dir}")
        sys.exit(1)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_images_dir)
                   if f.lower().endswith('.bmp')]

    for img_filename in image_files:
        img_path = os.path.join(input_images_dir, img_filename)

        # Corresponding label file (.tif)
        base_name = os.path.splitext(img_filename)[0]
        label_filename = base_name + '.tif'
        label_path = os.path.join(input_labels_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {img_filename}, skipping.")
            continue

        out_img_path = os.path.join(output_images_dir, img_filename)
        out_label_path = os.path.join(output_labels_dir, label_filename)

        # Crop and save
        crop_bottom(img_path, out_img_path, CROP_BOTTOM)
        crop_bottom(label_path, out_label_path, CROP_BOTTOM)
