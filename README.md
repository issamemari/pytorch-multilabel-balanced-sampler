# PyTorch Multilabel Balanced Samplers

This package provides samplers to fetch data samples from multilabel datasets in a balanced manner. Balanced sampling from multilabel datasets can be especially useful to handle class imbalance issues.

### Samplers

- `BaseMultilabelBalancedRandomSampler`: This is the base class for all the provided samplers. It initializes the basic structure required for sampling, such as class indices.

- `RandomClassSampler`: This sampler randomly chooses a class and then picks a random example from that class.

- `ClassCycleSampler`: As the name suggests, it cycles through each class and fetches a random example from the current class.

- `LeastSampledClassSampler`: Chooses the class with the least number of samples fetched so far and retrieves a random example from that class.

### Usage

#### Installation:

This package is installable via pip:

```shell
pip install pytorch-multilabel-balanced-sampler
```

#### Initialization:

For all samplers, the initialization arguments are:

- `labels`: A 2D tensor of shape `(n_examples, n_classes)` containing the one-hot encoded labels for the dataset.
- `indices`: A sequence of integers representing the indices of the dataset. Default is the range of the dataset size.

```python
from pytorch_multilabel_balanced_sampler.samplers import RandomClassSampler, ClassCycleSampler, LeastSampledClassSampler

sampler1 = RandomClassSampler(labels=my_labels, indices=my_indices)
sampler2 = ClassCycleSampler(labels=my_labels)
sampler3 = LeastSampledClassSampler(labels=my_labels, indices=my_indices)
```

#### Fetching samples:

Iterate over the sampler object to fetch samples:

```python
for sample in sampler1:
    print(sample)
```

### Note:

All samplers are inherited from `BaseMultilabelBalancedRandomSampler`, which in turn inherits from PyTorch's `Sampler` class. This ensures compatibility with PyTorch's data loading utilities.

### License

The MIT License (MIT). [License](https://github.com/issamemari/pytorch-multilabel-balanced-sampler/blob/master/LICENSE)

### Feedback & Issues

For feedback, issues, or feature requests, please raise an issue on the GitHub repository.
