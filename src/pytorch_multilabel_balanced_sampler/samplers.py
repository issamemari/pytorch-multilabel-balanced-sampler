import random
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data.sampler import Sampler

from .validation import validate_inputs


class BaseMultilabelBalancedSampler(Sampler[int]):
    def __init__(
        self, labels: torch.Tensor, indices: Optional[Sequence[int]] = None
    ) -> None:
        """
        Args:
            labels: A 2D tensor of shape (n_examples, n_classes) containing the one-hot
                encoded labels for the dataset.
        """
        validate_inputs(labels, indices)

        self.labels = labels
        self.num_classes = self.labels.shape[1]

        self.indices = indices if indices is not None else list(range(len(labels)))

        # Maps class index to indices of examples with that class
        self.class_indices = []
        for class_ in range(self.num_classes):
            self.class_indices.append(
                torch.nonzero(self.labels[self.indices, class_]).flatten().tolist()
            )

    def __iter__(self) -> Iterator[int]:
        self.count = 0
        return self

    def __next__(self) -> int:
        if self.count >= len(self.indices):
            raise StopIteration

        self.count += 1

        sample = self.sample()
        return self.indices[sample]

    def sample(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.indices)


class RandomClassSampler(BaseMultilabelBalancedSampler):
    """
    Chooses a class randomly, then chooses an example from that class randomly.
    """

    def sample(self) -> int:
        class_ = random.randint(0, self.num_classes - 1)
        return random.choice(self.class_indices[class_])


class ClassCycleSampler(BaseMultilabelBalancedSampler):
    """
    Cycles through classes, choosing randomly an example from each class in turn.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_class = 0

    def sample(self) -> int:
        class_ = self.current_class
        self.current_class = (self.current_class + 1) % self.num_classes
        return random.choice(self.class_indices[class_])


class LeastSampledClassSampler(BaseMultilabelBalancedSampler):
    """
    Chooses the class with the least number of samples so far, then chooses an example
    from that class randomly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = torch.zeros(self.num_classes, dtype=torch.long)

    def sample(self) -> int:
        classes = torch.nonzero(self.counts == self.counts.min(), as_tuple=True)[0]
        chosen_class = random.choice(classes.tolist())

        chosen_index = random.choice(self.class_indices[chosen_class])

        for class_, indicator in enumerate(self.labels[self.indices[chosen_index]]):
            if indicator == 1:
                self.counts[class_] += 1

        return chosen_index
