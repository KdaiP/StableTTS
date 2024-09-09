import torch

# copied from https://github.com/jaywalnut310/vits/blob/main/commons.py#L121
def sequence_mask(length: torch.Tensor, max_length: int = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)