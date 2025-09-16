import torch


class OptimizerFactory:
    """
    A factory class to create an optimizer with different parameter group strategies.
    This encapsulates the logic for setting up differential learning rates.
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.backbone_cutoff = 9  # Layers up to this index are considered the backbone

    def create(self, strategy: str):
        """
        Creates and returns an optimizer based on the specified strategy.

        Args:
            strategy (str): One of 'backbone_only', 'full_model_decay_user',
                            or 'full_model_decay_recommended'.
        """
        print(f"Creating optimizer with strategy: '{strategy}'")

        if strategy == 'backbone_only':
            parameter_groups = self._config_backbone_only()
        elif strategy == 'full_model_decay_user':
            parameter_groups = self._config_full_model_decay_user()
        elif strategy == 'full_model_decay_recommended':
            parameter_groups = self._config_full_model_decay_recommended()
        else:
            # Fallback to a default optimizer for the whole model
            parameter_groups = [{'params': self.model.parameters()}]

        return torch.optim.AdamW(parameter_groups, lr=self.args.lr0, momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)

    def _get_parameter_groups(self):
        """Helper method to separate model parameters into backbone and others."""
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True  # Ensure all parameters are trainable
            try:
                layer_index = int(name.split('.')[1])
                if layer_index <= self.backbone_cutoff:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            except (ValueError, IndexError):
                other_params.append(param)  # For params without a standard layer index
        return backbone_params, other_params

    def _config_backbone_only(self):
        """Freezes neck/head and prepares params for training only the backbone."""
        print("Strategy: Training backbone only. Freezing other layers.")
        for name, param in self.model.named_parameters():
            try:
                layer_index = int(name.split('.')[1])
                param.requires_grad = (layer_index <= self.backbone_cutoff)
            except (ValueError, IndexError):
                param.requires_grad = False
        return [{'params': [p for p in self.model.parameters() if p.requires_grad]}]

    def _config_full_model_decay_user(self):
        """Config for your request: higher LR for backbone, lower for the rest."""
        print("Strategy: High LR for backbone, low LR for neck/head.")
        backbone_params, other_params = self._get_parameter_groups()
        return [
            {'params': backbone_params, 'lr': self.args.lr0},
            {'params': other_params, 'lr': self.args.lr0 / 10.0}
        ]

    def _config_full_model_decay_recommended(self):
        """Config for the recommended approach: lower LR for backbone."""
        print("Strategy: Low LR for backbone, high LR for neck/head (Recommended).")
        backbone_params, other_params = self._get_parameter_groups()
        return [
            {'params': backbone_params, 'lr': self.args.lr0 / 10.0},
            {'params': other_params, 'lr': self.args.lr0}
        ]