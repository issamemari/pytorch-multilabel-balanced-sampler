from typing import Optional, Sequence

import torch


def validate_inputs(labels: torch.Tensor, indices: Optional[Sequence[int]] = None):
    """
    Validates the inputs to the samplers.
    """
    if not isinstance(labels, (torch.Tensor)):
        raise TypeError("labels must be a torch tensor")

    if labels.dtype not in (torch.int, torch.long):
        raise TypeError("labels must be a tensor of integers")

    if not len(labels.shape) == 2:
        raise ValueError("labels must be a 2D tensor")

    if not (labels == 0).logical_or(labels == 1).all():
        raise ValueError("labels must be one-hot encoded")

    if not labels.any(dim=0).all():
        raise ValueError("at least one example must exist for each class")

    if not isinstance(indices, (Sequence, type(None))):
        raise TypeError("indices must be an iterable or None")

    if indices is None:
        return

    if not all(isinstance(index, int) for index in indices):
        raise TypeError("indices must be a sequence of integers")

    if not all(0 <= index < len(labels) for index in indices):
        raise ValueError("indices must be in the range [0, len(labels))")
