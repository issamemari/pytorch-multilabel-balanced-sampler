import pytest
import torch

from pytorch_multilabel_balanced_sampler.validation import validate_inputs


def test_validate_inputs_valid():
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    indices = [0, 1, 2, 3]

    # Shouldn't raise any exceptions
    validate_inputs(labels, indices)


def test_validate_labels_not_tensor():
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]
    with pytest.raises(TypeError, match="labels must be a torch tensor"):
        validate_inputs(labels)


def test_validate_labels_wrong_dtype():
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(TypeError, match="labels must be a tensor of integers"):
        validate_inputs(labels)


def test_validate_labels_not_2d():
    labels = torch.tensor([1, 0, 1, 0])
    with pytest.raises(ValueError, match="labels must be a 2D tensor"):
        validate_inputs(labels)


def test_validate_labels_not_one_hot():
    labels = torch.tensor([[2, 0], [1, 0], [0, 1], [0, 1]])
    with pytest.raises(ValueError, match="labels must be one-hot encoded"):
        validate_inputs(labels)


def test_validate_labels_no_example_for_class():
    labels = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])
    with pytest.raises(
        ValueError, match="at least one example must exist for each class"
    ):
        validate_inputs(labels)


def test_validate_indices_wrong_type():
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    indices = "wrong_type"
    with pytest.raises(TypeError, match="indices must be a sequence of integers"):
        validate_inputs(labels, indices)


def test_validate_indices_non_integer():
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    indices = [0, 1, "2", 3]
    with pytest.raises(TypeError, match="indices must be a sequence of integers"):
        validate_inputs(labels, indices)


def test_validate_indices_out_of_range():
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    indices = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="indices must be in the range"):
        validate_inputs(labels, indices)
