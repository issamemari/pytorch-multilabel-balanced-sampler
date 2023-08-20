# PyTorch Multilabel Balanced Sampler

PyTorch sampler that outputs roughly balanced batches with support for multilabel datasets.

## How it works

Given a multilabel dataset of length n_samples and number of classes n_classes, samples from the data with equal probability per class, effectively oversampling minority classes and undersampling majority classes at the same time. Note that using this sampler does not guarantee that the distribution of classes in the output samples will be uniform, since the dataset is multilabel and sampling is based on a single class. It does however guarantee that all classes will have at least batch_size / n_classes samples as batch_size approaches infinity.


## License

The MIT License (MIT). [License](https://github.com/issamemari/pytorch-multilabel-balanced-sampler/blob/master/LICENSE)