import pytest
import torch

from pytorch_multilabel_balanced_sampler import (
    ClassCycleSampler,
    LeastSampledClassSampler,
    RandomClassSampler,
)


@pytest.mark.parametrize(
    "sampler", [RandomClassSampler, ClassCycleSampler, LeastSampledClassSampler]
)
def test_sampler(sampler):
    labels = torch.randint(0, 2, (100, 10))

    indices = torch.randperm(len(labels))[:50].tolist()

    samples = [next(iter(sampler(labels, indices))) for _ in range(len(indices))]

    assert all([sample in indices for sample in samples])
