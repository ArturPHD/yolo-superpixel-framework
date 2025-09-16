import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path

import sys

project_root = Path(__file__).resolve().parents[3]
yolov12_lib_path = project_root / "models" / "vendor" / "yolov12"
sys.path.insert(0, str(yolov12_lib_path))

from ultralytics.cfg import get_cfg
from models.custom.models.adjacency_channels.trainer import AdjacencyChannelsTrainer



def create_dataset_from_source(dataset_path: str, source_images_path: str, num_images: int = 5):
    """
    Creates a new dataset by copying a random subset of images
    from a source directory and generating dummy labels for them.

    Args:
        dataset_path (str): The path where the new dataset will be createde
        source_images_path (str): The path to the directory containing source images
                                  (e.g., E:\\MockDataset\\images).
        num_images (int): The number of random images to copy.
    """
    dataset_path = Path(dataset_path)
    source_images_path = Path(source_images_path)

    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    img_dir = dataset_path / 'train' / 'images'
    lbl_dir = dataset_path / 'train' / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for images in '{source_images_path}' and its subdirectories...")
    available_images = list(source_images_path.rglob('*.jpg')) + \
                       list(source_images_path.rglob('*.png')) + \
                       list(source_images_path.rglob('*.jpeg'))

    if not available_images:
        raise FileNotFoundError(f"No images (.jpg, .png, .jpeg) found in {source_images_path}")

    print(f"Found {len(available_images)} total images.")

    if num_images > len(available_images):
        print(
            f"Warning: Requested {num_images} images, but only {len(available_images)} are available. Using all available images.")
        num_images = len(available_images)

    selected_images = random.sample(available_images, num_images)
    print(f"Randomly selected {num_images} images to create the dataset.")

    for src_img_path in selected_images:
        dest_img_path = img_dir / src_img_path.name
        shutil.copy(src_img_path, dest_img_path)

        label_path = lbl_dir / f"{src_img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2")

    data_yaml_content = f"""
path: {dataset_path.resolve()}
train: train/images
val: train/images
nc: 1
names: ['object']
"""
    with open(dataset_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)

    print(f"\n✅ Dataset created successfully at '{dataset_path}'")
    return str(dataset_path / 'data.yaml')


def main():
    dataset_dir = Path.cwd() / "dummy_dataset_from_real_images"
    source_dir = r"E:\MockDataset\images"

    try:
        dummy_data_path = dummy_data_path = create_dataset_from_source(
            dataset_path=str(dataset_dir),
            source_images_path=source_dir,
            num_images=5
        )

        cfg = get_cfg()
        cfg.data = dummy_data_path
        cfg.model = 'yolov12.yaml'
        cfg.epochs = 3
        cfg.imgsz = 64
        cfg.batch = 16
        cfg.workers = 0
        cfg.plots = False
        cfg.val = True

        print("\nInitializing EmbedderYOLOTrainer...")
        trainer = AdjacencyChannelsTrainer(overrides=vars(cfg))
        print("Starting the training loop...")
        trainer.train()
        print("\n✅ Test passed! The Embedder architecture is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with an error: {e}")
        raise e
    finally:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"Cleaned up dummy dataset at {dataset_dir}.")

if __name__ == '__main__':
    main()