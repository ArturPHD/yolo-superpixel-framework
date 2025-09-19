import torch

from models.vendor.SPiT.spit.tokenizer.densesp import DenseSPEdgeEmbedder
from models.vendor.yolov12.ultralytics.models.yolo.detect import DetectionValidator
from models.vendor.yolov12.ultralytics.utils import ops


class AdjacencyChannelsYOLOModelValidator(DetectionValidator):
    """
    Custom Validator that handles 3-to-5 channel conversion for the validation loop.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        # Initialize the embedder for the validator
        self.edge_embedder = DenseSPEdgeEmbedder().to(self.device).eval()

    def preprocess(self, batch):
        """
        Overrides the preprocess method to convert 3-channel images to 5-channel.
        """
        # Let the parent class handle standard preprocessing
        batch = super().preprocess(batch)

        # Get the 3-channel image tensor
        rgb_batch = batch['img']

        # Use the embedder to convert it to a 5-channel tensor
        with torch.no_grad():
            five_channel_batch = self.edge_embedder(rgb_batch)

        # Replace the image tensor in the batch
        batch['img'] = five_channel_batch

        return batch

    # Your plotting methods can remain if you need them
    def _plot_batch(self, batch):
        plot_batch = batch.copy()
        plot_batch['img'] = batch['img'][:, :3, :, :].detach()
        return plot_batch

    def plot_val_samples(self, batch, ni):
        super().plot_val_samples(self._plot_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni):
        super().plot_predictions(self._plot_batch(batch), preds, ni)

    def update_metrics(self, preds, batch):
        """
        Flattens the target class tensor before metric calculation
        to prevent a NumPy ValueError.
        """
        # batch["cls"] = batch["cls"].flatten()
        super().update_metrics(preds, batch)
