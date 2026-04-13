import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from PIL import Image, ImageDraw


def make_mask(boxes, image_size):
    """
    Convert YOLO-format boxes to pixel masks.
    """
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        x_center, y_center, w, h = box

        # Convert from normalized to pixel coordinates
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        xmin = int(x_center - w / 2)
        ymin = int(y_center - h / 2)
        xmax = int(x_center + w / 2)
        ymax = int(y_center + h / 2)

        # Clamp values to image bounds (prevents errors)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        if xmax > xmin and ymax > ymin:
            draw.rectangle([xmin, ymin, xmax, ymax], fill=1)

    return np.array(mask, dtype=np.uint8)


def save_sample(image, mask, image_path, mask_path):
    image.save(image_path)
    Image.fromarray(mask * 255).save(mask_path)


def main():
    random.seed(42)

    output_root = Path("segmentation_dataset")
    images_root = output_root / "images"
    masks_root = output_root / "masks"

    for split in ["train", "val", "test"]:
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (masks_root / split).mkdir(parents=True, exist_ok=True)

    train_split = load_dataset("keremberke/satellite-building-segmentation", "mini", split="train")
    val_split = load_dataset("keremberke/satellite-building-segmentation", "mini", split="validation")
    test_split = load_dataset("keremberke/satellite-building-segmentation", "mini", split="test")

    dataset = concatenate_datasets([train_split, val_split, test_split])

    dataset = dataset.shuffle(seed=42)

    total = len(dataset)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_data = dataset.select(range(0, train_end))
    val_data = dataset.select(range(train_end, val_end))
    test_data = dataset.select(range(val_end, total))

    split_map = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split_name, split_data in split_map.items():
        for idx, sample in enumerate(split_data):
            image = sample["image"]

            objects = sample["objects"]
            boxes = objects["bbox"]

            mask = make_mask(boxes, image.size)

            image_path = images_root / split_name / f"sample_{idx:04d}.png"
            mask_path = masks_root / split_name / f"sample_{idx:04d}.png"

            save_sample(image, mask, image_path, mask_path)

    print("Dataset preparation complete.")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")


if __name__ == "__main__":
    main()