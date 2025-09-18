import shutil
import random
from pathlib import Path
import sys
import yaml

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
yolov12_lib_path = project_root / "models" / "vendor" / "yolov12"
sys.path.insert(0, str(yolov12_lib_path))

# --- Custom Imports ---
from models.custom.models.adjacency_channels.trainer import AdjacencyChannelsTrainer
from models.custom.models.adjacency_channels.validator import AdjacencyChannelsYOLOModelValidator
from models.vendor.yolov12.ultralytics.cfg import get_cfg


def create_dummy_dataset(dataset_path: Path, source_images_path: Path, num_images: int = 5):
    """Creates a small dummy dataset for testing."""
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    img_dir = dataset_path / 'train' / 'images'
    lbl_dir = dataset_path / 'train' / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    available_images = list(source_images_path.rglob('*.jpg')) + list(source_images_path.rglob('*.png'))
    selected_images = random.sample(available_images, min(num_images, len(available_images)))

    for src_img_path in selected_images:
        shutil.copy(src_img_path, img_dir / src_img_path.name)
        with open(lbl_dir / f"{src_img_path.stem}.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2")

    data_yaml_content = {
        'path': str(dataset_path.resolve()),
        'train': 'train/images',
        'val': 'train/images',
        'nc': 1,
        'names': ['object']
    }
    with open(dataset_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml_content, f)

    print(f"\n✅ Dummy dataset created at '{dataset_path}'")
    return dataset_path / 'data.yaml'


def main():
    """
    Runs a full end-to-end test: train, save, and then evaluate the saved model.
    """
    temp_dir = project_root / "temp_test_run"
    dataset_dir = temp_dir / "dummy_dataset"
    source_dir = project_root / "img"  # Assumes you have some test images in an 'img' folder

    run_name = "full_cycle_test"
    trained_model_path = None

    try:
        # --- STAGE 1: SETUP ---
        print("--- STAGE 1: SETUP ---")
        dummy_data_yaml_path = create_dummy_dataset(dataset_dir, source_dir, num_images=5)

        # --- STAGE 2: TRAINING ---
        print("\n--- STAGE 2: TRAINING ---")
        train_cfg = get_cfg(overrides={
            'model': 'yolov12n.yaml',
            'data': str(dummy_data_yaml_path),
            'epochs': 3,
            'imgsz': 64,
            'batch': 4,
            'workers': 0,
            'project': str(temp_dir / 'runs' / 'train'),
            'name': run_name,
            'val': True,
            'plots': False
        })

        trainer = AdjacencyChannelsTrainer(overrides=vars(train_cfg))
        trainer.train()

        trained_model_path = trainer.best
        print(f"\n✅ Training stage complete. Best model saved at: {trained_model_path}")

        # --- STAGE 3: EVALUATION ---
        print("\n--- STAGE 3: EVALUATION ---")
        if not trained_model_path.exists():
            raise FileNotFoundError("Best model was not saved after training.")

        eval_cfg = get_cfg(overrides={
            'model': str(trained_model_path),
            'data': str(dummy_data_yaml_path),
            'imgsz': 64,
            'batch': 4,
            'workers': 0,
            'save_json': True,
            'project': str(temp_dir / 'runs' / 'eval'),
            'name': run_name
        })

        validator = AdjacencyChannelsYOLOModelValidator(args=eval_cfg)
        validator()

        # Check if the JSON file was created
        json_path = validator.save_dir / "predictions.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Evaluation JSON file was not created at {json_path}")

        print(f"✅ Evaluation stage complete. Predictions saved to {json_path}")

        print("\n🎉🎉🎉 FULL TEST PASSED! 🎉🎉🎉")

    except Exception as e:
        print(f"\n❌ TEST FAILED with an error: {e}")
        raise e
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main()