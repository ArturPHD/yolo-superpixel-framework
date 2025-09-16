from models.vendor.yolov12.ultralytics.models.yolo.detect import DetectionValidator


class AdjacencyChannelsYOLOModelValidator(DetectionValidator):
    """
    Custom Validator for models with extra input channels.
    It extracts the first 3 (RGB) channels for visualization tasks.
    """
    def _plot_batch(self, batch):
        # This helper function ensures that plotting functions receive a 3-channel image
        plot_batch = batch.copy()
        plot_batch['img'] = batch['img'][:, :3, :, :].detach()

        print(f"Original image batch shape: {batch['img'].shape}")
        print(f"Plotting batch shape: {plot_batch['img'].shape}")
        print(f"Plotting batch requires grad: {plot_batch['img'].requires_grad}")

        return plot_batch

    def plot_val_samples(self, batch, ni):
        # Override to handle multi-channel validation samples
        super().plot_val_samples(self._plot_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni):
        # Override to handle multi-channel prediction plots
        super().plot_predictions(self._plot_batch(batch), preds, ni)