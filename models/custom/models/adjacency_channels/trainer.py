import torch
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.data import default_collate

from models.vendor.yolov12.ultralytics.models.yolo.detect import DetectionTrainer
from models.vendor.yolov12.ultralytics.data.utils import check_det_dataset
from models.vendor.yolov12.ultralytics.data.dataset import YOLODataset
from models.vendor.yolov12.ultralytics.data.build import InfiniteDataLoader

from models.vendor.SPiT.spit.tokenizer.densesp import DenseSPEdgeEmbedder

from .model import AdjacencyChannelsYOLOModel
from .validator import AdjacencyChannelsYOLOModelValidator
from .optimizer_factory import OptimizerFactory


class AdjacencyChannelsTrainer(DetectionTrainer):
    """
    Architecturally correct trainer.
    The data loader provides standard 3-channel images.
    The 3-to-5 channel conversion happens inside the main training loop.
    """

    def __init__(self, overrides=None, optimizer_strategy_config=None):
        super().__init__(overrides=overrides)
        self.optimizer_strategy_config = optimizer_strategy_config

        # Initialize the embedder and move it to the correct device
        self.edge_embedder = DenseSPEdgeEmbedder().to(self.device).eval()
        print("DenseSPEdgeEmbedder module is initialized and moved to the training device.")

    def preprocess_batch(self, batch):
        """
        Overrides the parent method to perform the 3-to-5 channel conversion.
        This method is called in every training step on the batch from the dataloader.
        """
        # 1. Let the parent class handle standard preprocessing (like moving to GPU)
        batch = super().preprocess_batch(batch)

        # 2. Get the 3-channel image tensor from the batch
        rgb_batch = batch['img']

        # 3. Use the embedder to convert it to a 5-channel tensor
        with torch.no_grad():
            five_channel_batch = self.edge_embedder(rgb_batch)

        # 4. Replace the image tensor in the batch with our new 5-channel version
        batch['img'] = five_channel_batch

        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Initializes and returns the AdjacencyChannelsYOLOModel."""
        model = AdjacencyChannelsYOLOModel(cfg or self.args.model, ch=3, nc=self.data['nc'], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a custom validator for this trainer."""
        # Note: The validator will also benefit from the overridden preprocess_batch
        return AdjacencyChannelsYOLOModelValidator(self.test_loader, save_dir=self.save_dir, args=self.args)

    def final_eval(self):
        """
        Overrides and disables the final evaluation step to prevent a crash
        when the framework tries to 'warm up' the custom 5-channel model
        with a standard 3-channel tensor. The best model is already saved.
        """
        print("Skipping final evaluation to avoid warmup bug with custom model. Best model is saved.")
        pass

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Overrides the default optimizer builder to use our OptimizerFactory.
        """
        if not self.optimizer_strategy_config:
            print("No optimizer strategy provided. Using default Ultralytics optimizer.")
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)

        print(f"Using OptimizerFactory to build optimizer with strategy: '{self.optimizer_strategy_config}'")
        factory = OptimizerFactory(model, self.args)
        return factory.create(strategy=self.optimizer_strategy_config)
