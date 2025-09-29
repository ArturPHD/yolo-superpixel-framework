import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from networkx.algorithms.centrality import local_reaching_centrality
import torch.distributed as dist

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
yolov12_lib_path = project_root / "models" / "vendor" / "yolov12"
sys.path.insert(0, str(yolov12_lib_path))

from models.custom.models.adjacency_channels.trainer import AdjacencyChannelsTrainer

TRAINER_REGISTRY = {
    'adjacency_channels': AdjacencyChannelsTrainer,
}


def find_config_file(config_identifier: str) -> Path:
    """Finds the config file, searching in the 'configs' directory if a direct path fails."""
    config_path = Path(config_identifier)
    if config_path.is_file():
        return config_path

    default_path = (project_root / 'configs' / config_identifier).with_suffix('.yaml')
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(f"Config '{config_identifier}' not found as a direct path or in 'configs/'.")



def setup_distributed(local_rank: int):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    """Loads a config file, prepares arguments, and runs the selected trainer."""
    try:
        using_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        rank = 0
        if using_ddp:
            rank, world_size = setup_distributed(args.local_rank)
            is_main = rank == 0
        else:
            is_main = True

        config_path = find_config_file(args.config)
        print(f"Loading configuration from: {config_path}")

        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                print("CUDA is available. Using GPU for training.")
            else:
                print(f"CUDA is available. Using {torch.cuda.device_count()} GPUs for training:")
            for i in range(torch.cuda.device_count()):
                print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        implementation_name = config.pop('implementation_name')
        TrainerClass = TRAINER_REGISTRY[implementation_name]

        model_cfg = config.pop('model_config')
        config['model'] = f"yolov12{model_cfg['scale']}.yaml"

        config['project'] = str(project_root / 'runs' / implementation_name)

        optimizer_strategy_config = config.pop('optimizer_strategy', None)

        print(f"\nInitializing trainer for '{implementation_name}'...")
        trainer = TrainerClass(
            overrides=config,
            optimizer_strategy_config=optimizer_strategy_config,
            local_rank=args.local_rank if using_ddp else 0,
            world_size=world_size if using_ddp else 1
        )

        trainer.train()
        print("\n✅ Training finished successfully!")

    except Exception as e:
        print(f"\n❌ Training failed with an error: {e}")
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO training from a config file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Name of the experiment config file (e.g., my_experiment_v1).')
    args = parser.parse_args()
    main(args)