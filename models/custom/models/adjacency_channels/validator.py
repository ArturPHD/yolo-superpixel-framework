import torch
from models.vendor.yolov12.ultralytics.models.yolo.detect import DetectionValidator
from models.vendor.yolov12.ultralytics.utils import ops

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

    def _prepare_batch(self, si, batch):
        """
        Prepares a single sample from a batch for metric calculation.
        This method completely overrides the buggy parent method.
        """
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx]
        bbox = batch['bboxes'][idx]
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        ratio_pad = batch['ratio_pad'][si]

        # This check is now the first thing we do, preventing crashes.
        if cls.numel() > 0:
            if ratio_pad:
                # Denormalize ground-truth bboxes
                bbox[:, :4] = ops.xywhn2xyxy(x=bbox[:, :4], w=imgsz[1], h=imgsz[0], padw=ratio_pad[1][0], padh=ratio_pad[1][1])

        # We return a new dictionary containing all necessary keys.
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
