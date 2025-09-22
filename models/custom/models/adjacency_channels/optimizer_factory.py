import torch


class OptimizerFactory:
    """
    A factory class to create an optimizer with differential learning rates.
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args

    def create(self, strategy: dict):
        """
        Creates and returns an optimizer based on the specified strategy dictionary.

        Args:
            strategy (dict): A dictionary defining the optimizer strategy.
        """
        if not isinstance(strategy, dict) or 'type' not in strategy:
            print("Warning: Invalid or missing optimizer strategy. Falling back to a default optimizer.")
            parameter_groups = [{'params': self.model.parameters()}]

        else:
            strategy_type = strategy.get('type')
            print(f"Creating optimizer with strategy type: '{strategy_type}'")

            if strategy_type == 'differential_lr':
                parameter_groups = self._config_differential_lr(strategy.get('groups', []))
            else:
                print(f"Warning: Unknown strategy type '{strategy_type}'. Falling back to default.")
                parameter_groups = [{'params': self.model.parameters()}]

        return torch.optim.AdamW(parameter_groups, lr=self.args.lr0, weight_decay=self.args.weight_decay)

    def _config_differential_lr(self, groups_config: list):
        """
        Configures parameter groups with differential learning rates.
        """
        if not groups_config:
            print("Warning: 'differential_lr' strategy selected, but no groups are defined. Using default.")
            return [{'params': self.model.parameters()}]

        print("Configuring optimizer with differential learning rates.")

        groups_config = sorted(groups_config, key=lambda x: x['layer_cutoff'])
        param_groups = {group['name']: [] for group in groups_config}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            try:
                # e.g., 'model.0.conv.weight' -> layer_index = 0
                layer_index = int(name.split('.')[1])

                assigned = False
                for group in groups_config:
                    if layer_index <= group['layer_cutoff']:
                        param_groups[group['name']].append(param)
                        assigned = True
                        break
                if not assigned:
                    param_groups[groups_config[-1]['name']].append(param)

            except (ValueError, IndexError):
                # For parameters without a standard layer index, assign to the last group
                param_groups[groups_config[-1]['name']].append(param)

        final_optimizer_groups = []
        base_lr = self.args.lr0

        for group_config in groups_config:
            group_name = group_config['name']
            params_list = param_groups[group_name]

            if not params_list:
                print(f"Info: Group '{group_name}' has 0 parameters. Skipping.")
                continue

            lr_multiplier = group_config['lr_multiplier']
            group_lr = base_lr * lr_multiplier

            final_optimizer_groups.append({
                'params': params_list,
                'lr': group_lr
            })
            print(f"  - Group '{group_name}': {len(params_list)} params, LR = {group_lr:.1e}")

        return final_optimizer_groups