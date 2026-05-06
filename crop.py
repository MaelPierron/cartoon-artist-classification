import os
from PIL import Image

def split_into_two_squares(image_path, output_dir):
    img = Image.open(image_path)
    width, height = img.size

    square_size = min(width, height)

    # Cas 1 : image plus large que haute → gauche + droite
    if width > height:
        left_box = (0, 0, square_size, square_size)
        right_box = (width - square_size, 0, width, square_size)

        square1 = img.crop(left_box)
        square2 = img.crop(right_box)

    # Cas 2 : image plus haute que large → haut + bas
    else:
        top_box = (0, 0, square_size, square_size)
        bottom_box = (0, height - square_size, square_size, height)

        square1 = img.crop(top_box)
        square2 = img.crop(bottom_box)

    # Sauvegarde
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if square1.mode == 'RGBA':
        square1 = square1.convert('RGB')
    if square2.mode == 'RGBA':
        square2 = square2.convert('RGB')

    square_size_output = (384, 384)

    square1 = square1.resize(square_size_output, Image.Resampling.LANCZOS)
    square2 = square2.resize(square_size_output, Image.Resampling.LANCZOS)
    square1.save(os.path.join(output_dir, f"{base_name}_square1.jpg"))
    square2.save(os.path.join(output_dir, f"{base_name}_square2.jpg"))


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(input_dir, filename)
            split_into_two_squares(path, output_dir)



input_folder = "data/input_dataset/One_piece"
output_folder = "dataset/train/One_piece"

process_folder(input_folder, output_folder)