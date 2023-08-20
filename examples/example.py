import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch_multilabel_balanced_sampler import (
    LeastSampledClassSampler,
    RandomClassSampler,
)


class RandomMultilabelDataset(Dataset):
    def __init__(self, *, n_examples, n_classes, mean_labels_per_example):
        class_probabilities = torch.rand([n_classes])
        class_probabilities = class_probabilities / sum(class_probabilities)
        class_probabilities *= mean_labels_per_example
        self.y = (torch.rand([n_examples, n_classes]) < class_probabilities).int()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"labels": Variable(torch.tensor(self.y[index]), requires_grad=False)}


def get_data_loaders(batch_size, val_size):
    dataset = RandomMultilabelDataset(
        n_examples=100000,
        n_classes=20,
        mean_labels_per_example=2,
    )

    # Split into training and validation
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * len(dataset)))
    train_idx, validate_idx = indices[split:], indices[:split]

    train_sampler = RandomClassSampler(dataset.y, train_idx)
    validate_sampler = SubsetRandomSampler(validate_idx)

    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    validate_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=validate_sampler,
    )
    return train_loader, validate_loader


def main():
    epochs = 2
    train_loader, validate_loader = get_data_loaders(batch_size=512, val_size=0.5)

    for _ in range(epochs):
        print("================ Training phase ===============")
        for batch in train_loader:
            labels = batch["labels"]
            print("Label counts per class:")
            sum_ = labels.sum(axis=0)
            print(sum_)
            print("Difference between min and max")
            print(max(sum_) - min(sum_))
            print("")
        print("")

        print("=============== Validation phase ==============")
        for batch in validate_loader:
            labels = batch["labels"]
            print("Label counts per class:")
            sum_ = labels.sum(axis=0)
            print(sum_)
            print("Difference between min and max")
            print(max(sum_) - min(sum_))
            print("")
        print("")


if __name__ == "__main__":
    main()
