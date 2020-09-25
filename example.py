import torch
import numpy as np

from sampler import MultilabelBalancedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class RandomDataset(Dataset):
    def __init__(self, n_examples, n_features, n_classes, mean_labels_per_example):
        self.n_examples = n_examples
        self.n_features = n_features
        self.n_classes = n_classes
        self.X = np.random.random([self.n_examples, self.n_features])

        class_probabilities = np.random.random([self.n_classes])
        class_probabilities = class_probabilities / sum(class_probabilities)
        class_probabilities *= mean_labels_per_example
        self.y = (
            np.random.random([self.n_examples, self.n_classes]) < class_probabilities
        ).astype(int)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = Variable(torch.tensor(self.X[index]), requires_grad=False)
        labels = Variable(torch.tensor(self.y[index]), requires_grad=False)
        return {"example": example, "labels": labels}


def get_data_loaders(batch_size, val_size):
    dataset = RandomDataset(20000, 100, 20, 2)

    # Split into training and validation
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * len(dataset)))
    train_idx, validate_idx = indices[split:], indices[:split]

    train_sampler = MultilabelBalancedRandomSampler(
        dataset.y, train_idx, class_choice="least_sampled"
    )
    validate_sampler = SubsetRandomSampler(validate_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,)
    validate_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=validate_sampler,
    )
    return train_loader, validate_loader


def main():
    epochs = 2
    train_loader, validate_loader = get_data_loaders(batch_size=512, val_size=0.2)

    for epoch in range(epochs):
        print("================ Training phase ===============")
        for batch in train_loader:
            examples = batch["example"]
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
            examples = batch["example"]
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
