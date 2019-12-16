import torch
import numpy as np

from sampler import MultilabelBalancedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class RandomDataset(Dataset):
    def __init__(self, n_samples, n_features, n_labels, labels_only=False):
        self.labels_only = labels_only
        self.n_samples = 1000
        self.n_features = 100
        self.n_classes = 10
        self.X = np.random.random([self.n_samples, self.n_features])
        self.y = (np.random.random([self.n_samples, self.n_classes]) > 0.5).astype(int)

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        example = Variable(torch.tensor(self.X[index]), requires_grad=False)
        labels = Variable(torch.tensor(self.y[index]), requires_grad=False)
        return {"example": example, "labels": labels}


def get_data_loaders(batch_size, val_size):
    dataset = RandomDataset(1000, 100, 10)

    # Split into training and validation
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * len(dataset)))
    train_idx, validate_idx = indices[split:], indices[:split]

    train_sampler = MultilabelBalancedRandomSampler(dataset.y, train_idx)
    validate_sampler = SubsetRandomSampler(validate_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,)
    validate_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=validate_sampler,
    )
    return train_loader, validate_loader


def main():
    epochs = 10
    train_loader, validate_loader = get_data_loaders(batch_size=64, val_size=0.2)

    for epoch in range(epochs):
        print("Training phase")
        for batch in train_loader:
            examples = batch["example"]
            labels = batch["labels"]
            print("Examples: {}".format(examples))
            print("Labels: {}".format(labels))

        print("Validation phase")
        for batch in validate_loader:
            examples = batch["example"]
            labels = batch["labels"]
            print("Examples: {}".format(examples))
            print("Labels: {}".format(labels))


if __name__ == "__main__":
    main()
