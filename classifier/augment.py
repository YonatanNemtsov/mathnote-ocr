#!/usr/bin/env python3
"""
Data augmentation for symbol training images.
Generates augmented copies with rotation, scale, stroke jitter, and noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from PIL import Image, ImageFilter, ImageTransform
import math

import config


def augment_image(
    img: Image.Image,
    rotation_range: float = 15.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    squeeze_range: float = 0.30,
    translate_range: float = 5.0,
    add_noise: bool = True,
) -> Image.Image:
    """
    Apply random augmentation to a 128×128 grayscale symbol image.
    """
    size = img.size[0]

    # Random rotation
    angle = random.uniform(-rotation_range, rotation_range)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # Random squeeze (independent x/y scaling)
    sx = random.uniform(1.0 - squeeze_range, 1.0 + squeeze_range)
    sy = random.uniform(1.0 - squeeze_range, 1.0 + squeeze_range)
    new_w = max(1, int(size * sx))
    new_h = max(1, int(size * sy))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Crop or pad back to original size
    result = Image.new("L", (size, size), 255)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    # Paste (clipping handled by paste coordinates)
    paste_x = max(ox, 0)
    paste_y = max(oy, 0)
    crop_x = max(-ox, 0)
    crop_y = max(-oy, 0)
    crop_w = min(new_w, size - paste_x + crop_x) - crop_x
    crop_h = min(new_h, size - paste_y + crop_y) - crop_y
    if crop_w > 0 and crop_h > 0:
        cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        result.paste(cropped, (paste_x, paste_y))
    img = result

    # Random translation
    tx = random.uniform(-translate_range, translate_range)
    ty = random.uniform(-translate_range, translate_range)
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, -tx, 0, 1, -ty),
        resample=Image.BILINEAR,
        fillcolor=255,
    )

    # Slight blur for stroke thickness variation
    if add_noise and random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


def augment_dataset(
    source_dir: Path = config.SYMBOLS_DIR,
    output_dir: Path = config.AUGMENTED_DIR,
    copies_per_image: int = 3,
):
    """
    Generate augmented copies of all training images.

    Creates `copies_per_image` augmented versions of each image,
    saved to output_dir with the same class directory structure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    total_generated = 0

    for symbol_dir in symbol_dirs:
        class_name = symbol_dir.name
        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(exist_ok=True)

        images = sorted(symbol_dir.glob("*.png"))

        for img_path in images:
            img = Image.open(img_path).convert("L")

            for i in range(copies_per_image):
                aug = augment_image(img)
                out_name = f"{img_path.stem}_aug{i:02d}.png"
                aug.save(out_class_dir / out_name)
                total_generated += 1

        count = len(images)
        print(f"  {class_name}: {count} originals → {count * copies_per_image} augmented")

    print(f"\nTotal augmented images: {total_generated}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    print("Generating augmented training data...\n")
    augment_dataset()
