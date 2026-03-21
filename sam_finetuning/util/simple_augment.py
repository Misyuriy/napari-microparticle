import os
import sys
import glob
import argparse
from PIL import Image


def apply_transform(img, angle, flip_h):
    # expand=False keeps original dimensions
    if angle != 0:
        img = img.rotate(angle, expand=False, fillcolor=0)

    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


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

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    # Check folders exist
    if not os.path.isdir(images_dir):
        print(f"Error: '{images_dir}' folder not found.")
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print(f"Error: '{labels_dir}' folder not found.")
        sys.exit(1)

    image_paths = glob.glob(os.path.join(images_dir, "*.bmp"))

    transforms = [
        #(0, False),  # identity
        (90, False),
        (180, False),
        (270, False),
        (0, True),
        (90, True),
        (180, True),
        (270, True),
    ]

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(labels_dir, base + ".tif")

        if not os.path.isfile(mask_path):
            print(f"Warning: No matching mask for {img_path}, skipping.")
            continue

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        for idx, (angle, flip_h) in enumerate(transforms, start=1):
            img_aug = apply_transform(img, angle, flip_h)
            mask_aug = apply_transform(mask, angle, flip_h)

            img_out = os.path.join(images_dir, f"{base}_aug{idx}.bmp")
            mask_out = os.path.join(labels_dir, f"{base}_aug{idx}.tif")

            img_aug.save(img_out)
            mask_aug.save(mask_out)

        print(f"Processed {base}")
