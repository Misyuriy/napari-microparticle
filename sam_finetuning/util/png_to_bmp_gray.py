import sys
from PIL import Image


def png_to_single_channel_bmp(input_path, output_path):
    try:
        # Open the input image
        with Image.open(input_path) as img:
            # Convert to single-channel grayscale
            gray_img = img.convert('L')
            # Save as BMP (8-bit grayscale)
            gray_img.save(output_path, format='BMP')
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python png_to_bmp_gray.py input.png output.bmp")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    png_to_single_channel_bmp(input_file, output_file)
