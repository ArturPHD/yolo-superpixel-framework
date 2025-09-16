import torch
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.data import default_collate

# Use absolute imports for clarity and robustness
from models.vendor.yolov12.ultralytics.models.yolo.detect import DetectionTrainer
from models.vendor.yolov12.ultralytics.data.utils import check_det_dataset
from models.vendor.yolov12.ultralytics.data.dataset import YOLODataset
from models.vendor.yolov12.ultralytics.data.build import InfiniteDataLoader

from models.vendor.SPiT.spit.tokenizer.densesp import DenseSPEdgeEmbedder

# Use relative imports for local package modules
from .model import AdjacencyChannelsYOLOModel
from .validator import AdjacencyChannelsYOLOModelValidator


class AdjacencyChannelsTrainer(DetectionTrainer):
    """
    Final, robust trainer version that correctly handles the entire data pipeline.
    """

    def __init__(self, overrides=None):
        super().__init__(overrides=overrides)
        self.edge_embedder = None
        self._init_embedder()

    def _init_embedder(self):
        print("Initializing DenseSPEdgeEmbedder module...")
        self.edge_embedder = DenseSPEdgeEmbedder()
        self.edge_embedder = self.edge_embedder.to(self.device).train()
        print("Embedder module is ready.")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode=mode)

        # Apply our custom collate function to both 'train' and 'val' modes.
        # This ensures that the validation data is processed correctly.
        collate_fn = self.embedder_collate if mode in ('train', 'val') else torch.utils.data.default_collate

        pin_memory_flag = False

        if self.args.workers > 0:
            return InfiniteDataLoader(dataset=dataset, batch_size=batch_size, shuffle=mode == 'train',
                                      num_workers=self.args.workers, sampler=None, collate_fn=collate_fn,
                                      pin_memory=pin_memory_flag, rank=rank, mode=mode)
        else:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=mode == 'train',
                              num_workers=0, sampler=None, collate_fn=collate_fn, pin_memory=pin_memory_flag)

    def embedder_collate(self, batch):
        """
        Correctly constructs the batch dictionary for the Ultralytics loss function,
        including separate tensors for batch_idx, cls, and bboxes.
        """
        # 1. Process images
        images = torch.stack([b['img'] for b in batch])
        rgb_batch_float = images.float() / 255.0
        device = next(self.edge_embedder.parameters()).device
        rgb_batch_gpu = rgb_batch_float.to(device)
        final_5_channel_batch = self.edge_embedder(rgb_batch_gpu)
        print(f"Shape of 5-channel batch: {final_5_channel_batch.shape}")

        # 2. Manually process labels and image paths
        batch_idx_list = []
        cls_list = []
        bboxes_list = []
        im_file_list = []

        print(f"Number of samples in batch: {len(batch)}")

        for i, sample in enumerate(batch):
            cls_tensor = sample.get('cls')
            bboxes_tensor = sample.get('bboxes')
            im_file = sample.get('im_file')

            if cls_tensor is not None and bboxes_tensor is not None and len(cls_tensor) > 0:
                batch_idx_list.append(torch.full((len(cls_tensor), 1), i))
                cls_list.append(cls_tensor)
                bboxes_list.append(bboxes_tensor)

            if im_file is not None:
                im_file_list.append(im_file)

        print(f"Number of detected objects in batch: {len(batch_idx_list)}")

        # 3. Assemble the final batch dictionary with correct keys
        collated_batch = {}
        collated_batch['img'] = final_5_channel_batch.detach().cpu()
        collated_batch['im_file'] = im_file_list

        if batch_idx_list:
            collated_batch['batch_idx'] = torch.cat(batch_idx_list, 0)
            collated_batch['cls'] = torch.cat(cls_list, 0)
            collated_batch['bboxes'] = torch.cat(bboxes_list, 0)
        else:
            collated_batch['batch_idx'] = torch.empty(0)
            collated_batch['cls'] = torch.empty(0)
            collated_batch['bboxes'] = torch.empty(0, 4)

        print(f"Collated batch keys: {collated_batch.keys()}")
        print(f"Shape of 'img' tensor: {collated_batch['img'].shape}")
        print(f"Shape of 'batch_idx' tensor: {collated_batch['batch_idx'].shape}")
        print(f"Shape of 'cls' tensor: {collated_batch['cls'].shape}")
        print(f"Shape of 'bboxes' tensor: {collated_batch['bboxes'].shape}")

        collated_batch['batch_idx'] = collated_batch['batch_idx'].squeeze()
        collated_batch['cls'] = collated_batch['cls'].squeeze()

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
            # If validation is enabled in the config, run the real validator.
            print("Validation is enabled, running the standard validator...")
            return super().validate()
        else:
            # If validation is disabled, skip it and return empty metrics.
            print("Validation is disabled (val=False), skipping validation step.")
            return {}, 0.0

    def final_eval(self):
        """
        Conditionally overrides the final evaluation method.

        If 'val=False' is set in the config, it skips the final evaluation
        to prevent errors during testing.

        If 'val=True', it calls the original final_eval method from the
        parent class to run a full evaluation on the best saved model.
        """
        if self.args.val:
            # If validation is enabled, run the original final evaluation.
            print("Validation is enabled, running final evaluation on the best model...")
            super().final_eval()
        else:
            # If validation is disabled, just print a message and do nothing.
            print("Validation is disabled (val=False), skipping final evaluation.")