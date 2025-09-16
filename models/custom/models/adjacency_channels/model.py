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
        return self.act(self.bn(self.conv(x)))


class AdjacencyChannelsYOLOModel(DetectionModel):
    def __init__(self, cfg='yolov12.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

        first_layer = self.model[0]
        if not isinstance(first_layer, Conv):
            raise TypeError("The model's first layer is not a standard ultralytics Conv block.")

        old_conv = first_layer.conv

        if old_conv.in_channels == 5:
            print("Model is already 5-channel. Skipping modification.")
            return

        print("Patching the first Conv layer to accept 5-channel input while preserving RGB weights.")

        # Create a new 5-channel convolutional layer
        new_conv = nn.Conv2d(
            in_channels=5,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = old_conv.weight.data.clone()
            new_conv.weight.data[:, 3:, :, :] = torch.randn_like(new_conv.weight.data[:, 3:, :, :]) * 1e-6

            if new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

        first_layer.conv = new_conv

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Overrides the base warmup method to use a 5-channel input tensor,
        preventing a channel mismatch error during validation or export.
        """
        if self.device.type != 'cpu':
            im = torch.zeros(imgsz[0], 5, imgsz[2], imgsz[3], dtype=self.dtype, device=self.device)
            self.forward(im)
        return self