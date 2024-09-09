import torch
import torch.nn as nn

from .backbone import ConvNeXtEncoder
from .head import HiFiGANGenerator

config_dict = {
    "backbone": {
        # "input_channels": "${model.num_mels}",
        "input_channels": 128,
        "depths": [3, 3, 9, 3],
        "dims": [128, 256, 384, 512],
        "drop_path_rate": 0.2,
        "kernel_size": 7,
    },
    "head": {
        # "hop_length": "${model.hop_length}",
        "hop_length": 512,
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "num_mels": 512,  # consistent with the output of the backbone
        "upsample_initial_channel": 512,
        "use_template": False,
        "pre_conv_kernel_size": 13,
        "post_conv_kernel_size": 13,
    }
}

# download_link: https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt
class FireflyGANBaseWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = FireflyGANBase()
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        
        self.model.eval()

    @ torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FireflyGANBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtEncoder(**config_dict["backbone"])
        self.head = HiFiGANGenerator(**config_dict["head"])
        
        self.head.checkpointing = False

    @ torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x.squeeze(1)