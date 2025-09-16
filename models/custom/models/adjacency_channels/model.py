import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv


class LazyConv(nn.Module):
    def __init__(self, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.LazyConv2d(c2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """Forward pass through the LazyConv block."""
        return self.act(self.bn(self.conv(x)))


class AdjacencyChannelsYOLOModel(DetectionModel):
    """
    A custom YOLO model where the first convolutional layer is replaced by our LazyConv.
    """

    def __init__(self, cfg='yolov12.yaml', ch=3, nc=None, verbose=True, zero_new_weights=False):
        super().__init__(cfg, ch, nc, verbose)

        first_layer = self.model[0]

        if isinstance(first_layer, Conv):
            print("Replacing the first Conv layer with a custom initialized Conv2d layer.")

            new_conv = nn.Conv2d(
                in_channels=5,
                out_channels=first_layer.conv.out_channels,
                kernel_size=first_layer.conv.kernel_size,
                stride=first_layer.conv.stride,
                padding=first_layer.conv.padding,
                bias=False
            )

            original_weights = torch.randn(
                first_layer.conv.out_channels,
                3,
                *first_layer.conv.kernel_size
            )

            print(f"Original 3-channel weights shape: {original_weights.shape}")

            if zero_new_weights:
                print("Initializing new channel weights to zero.")
                new_weights = torch.zeros(
                    first_layer.conv.out_channels,
                    2,  # New 2 channels
                    *first_layer.conv.kernel_size
                )
            else:
                print("Initializing new channel weights with small random values.")
                new_weights = torch.randn(
                    first_layer.conv.out_channels,
                    2,
                    *first_layer.conv.kernel_size
                ) * 1e-6

            print(f"New 2-channel weights shape: {new_weights.shape}")
            print(f"Sum of new channel weights (should be close to 0 if zeroed): {new_weights.sum()}")

            combined_weights = torch.cat((original_weights, new_weights), dim=1)

            new_conv.weight.data.copy_(combined_weights)

            new_conv_block = Conv(
                c1=5,
                c2=first_layer.conv.out_channels,
                k=first_layer.conv.kernel_size,
                s=first_layer.conv.stride,
                p=first_layer.conv.padding,
                act=first_layer.act
            )

            new_conv_block.conv = new_conv
            new_conv_block.bn = first_layer.bn
            new_conv_block.act = first_layer.act

            new_conv_block.i = first_layer.i
            new_conv_block.f = first_layer.f

            self.model[0] = new_conv_block
        else:
            raise TypeError("The model's first layer is not a standard ultralytics Conv block.")

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Overrides the base warmup method to use a 5-channel input tensor,
        preventing a channel mismatch error during validation or export.
        """
        # The device and dtype are attributes of the parent BaseModel class
        if self.device.type != 'cpu':
            # Create a dummy input tensor with the correct number of channels (5)
            # Ignore the '3' from the input `imgsz` tuple and hardcode 5.
            im = torch.zeros(imgsz[0], 5, imgsz[2], imgsz[3], dtype=self.dtype, device=self.device)

            # Perform a forward pass to warm up the model
            self.forward(im)
        return self