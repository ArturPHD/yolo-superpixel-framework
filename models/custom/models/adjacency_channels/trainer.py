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
    """Custom trainer for the Adjacency Channels model."""
    def __init__(self, overrides=None, optimizer_strategy_config=None):
        super().__init__(overrides=overrides)
        self.optimizer_strategy_config = optimizer_strategy_config
        self.edge_embedder = None
        self._init_embedder()

    def _init_embedder(self):
        print("Initializing DenseSPEdgeEmbedder module...")
        self.edge_embedder = DenseSPEdgeEmbedder()
        self.edge_embedder.eval()
        print("Embedder module is ready.")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Builds and returns a DataLoader for a given dataset."""
        assert mode in ['train', 'val']
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)
        collate_fn = self.embedder_collate

        if mode == 'train':
            if self.args.workers > 0:
                return InfiniteDataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=self.args.workers,
                                          sampler=None,
                                          collate_fn=collate_fn,
                                          pin_memory=True)
            else:
                return DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  sampler=None,
                                  collate_fn=collate_fn,
                                  pin_memory=False)
        else:
            return DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=self.args.workers,
                              sampler=None,
                              collate_fn=collate_fn,
                              pin_memory=True)

    def setup_optimizer(self):
        """Initializes the optimizer using the OptimizerFactory."""
        factory = OptimizerFactory(self.model, self.args)
        # Pass the entire config dictionary to the factory
        return factory.create(config=self.optimizer_strategy_config)

    def embedder_collate(self, batch):
        """
        A robust collate function that handles variable numbers of objects,
        preserves all necessary metadata, and gracefully handles missing keys
        by providing default values.
        """
        images = torch.stack([b['img'] for b in batch])
        rgb_batch_float = images.float() / 255.0
        device = next(self.edge_embedder.parameters()).device

        with torch.no_grad():
            final_5_channel_batch = self.edge_embedder(rgb_batch_float)

        batch_idx_list, cls_list, bboxes_list = [], [], []

        other_keys = {
            'im_file': [],
            'ori_shape': [],
            'resized_shape': [],
            'ratio_pad': []
        }

        for i, sample in enumerate(batch):
            # Handle labels
            cls_tensor, bboxes_tensor = sample.get('cls'), sample.get('bboxes')
            if cls_tensor is not None and bboxes_tensor is not None and len(cls_tensor) > 0:
                batch_idx_list.append(torch.full((len(cls_tensor), 1), i, device=cls_tensor.device))
                cls_list.append(cls_tensor)
                bboxes_list.append(bboxes_tensor)

            # Use .get() to provide a default value (None) if a key is missing.
            # This ensures all metadata lists have the same length as the number of images.
            for key in other_keys:
                other_keys[key].append(sample.get(key, None))

        collated_batch = {'img': final_5_channel_batch.detach()}

        if batch_idx_list:
            collated_batch['batch_idx'] = torch.cat(batch_idx_list, 0).squeeze(-1)
            collated_batch['cls'] = torch.cat(cls_list, 0).squeeze(-1)
            collated_batch['bboxes'] = torch.cat(bboxes_list, 0)
        else:
            collated_batch['batch_idx'] = torch.empty(0)
            collated_batch['cls'] = torch.empty(0)
            collated_batch['bboxes'] = torch.empty(0, 4)

        collated_batch.update(other_keys)

        return collated_batch

    def build_dataset(self, img_path, mode='train', batch=None):
        data_cfg = check_det_dataset(self.args.data)
        return YOLODataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                           stride=32, data=data_cfg)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = AdjacencyChannelsYOLOModel(cfg or self.args.model, ch=3, nc=self.data['nc'], verbose=verbose)
        model.to(self.device)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return AdjacencyChannelsYOLOModelValidator(self.test_loader, save_dir=self.save_dir, args=self.args)

    def validate(self):
        """
        Conditionally overrides the validation method.

        If 'val=False' is set in the config (e.g., during a quick test),
        it skips the validation step to prevent errors on dummy data.

        If 'val=True' (during actual training), it calls the original
        validation method from the parent class to run a full evaluation.
        """
        if self.args.val:
            print("Validation is enabled, running the standard validator...")
            return super().validate()
        else:
            print("Validation is disabled (val=False), skipping validation step.")
            return {}, 0.0

    def final_eval(self):
        """
        Overrides and disables the final evaluation step to prevent a crash
        when the framework tries to 'warm up' the custom 5-channel model
        with a standard 3-channel tensor. The best model is already saved.
        """
        print("Skipping final evaluation to avoid warmup bug with custom model. Best model is saved.")
        pass