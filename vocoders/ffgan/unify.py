import torch
from torch import nn


class UnifyGenerator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        vq: nn.Module | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.vq = vq

    def forward(self, x: torch.Tensor, template=None) -> torch.Tensor:
        x = self.backbone(x)

        if self.vq is not None:
            vq_result = self.vq(x)
            x = vq_result.z

        x = self.head(x, template=template)

        if x.ndim == 2:
            x = x[:, None, :]

        if self.vq is not None:
            return x, vq_result

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.vq is None:
            raise ValueError("VQ module is not present in the model.")

        x = self.backbone(x)
        vq_result = self.vq(x)
        return vq_result.codes

    def decode(self, codes: torch.Tensor, template=None) -> torch.Tensor:
        if self.vq is None:
            raise ValueError("VQ module is not present in the model.")

        x = self.vq.from_codes(codes)[0]
        x = self.head(x, template=template)

        if x.ndim == 2:
            x = x[:, None, :]

        return x

    def remove_parametrizations(self):
        if hasattr(self.backbone, "remove_parametrizations"):
            self.backbone.remove_parametrizations()

        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()