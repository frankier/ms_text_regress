"""
Taken from pytorch-metric-learning and modified to output lists of ints rather than numpy arrays.

Originally Apache-v2 licensed.

https://github.com/KevinMusgrave/pytorch-metric-learning/blob/345cb2b0cabfd39cc7a25ff2a81fa29adc3ea885/src/pytorch_metric_learning/samplers/hierarchical_sampler.py
"""

import itertools
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler


def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)


# Inspired by
# https://github.com/kunhe/Deep-Metric-Learning-Baselines/blob/master/datasets.py
class HierarchicalSampler(BatchSampler):
    def __init__(
        self,
        labels,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to "all" if all element in a class
        batches_per_super_tuples: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        self.batch_size = batch_size
        self.batches_per_super_tuple = batches_per_super_tuple
        self.samples_per_class = samples_per_class
        self.super_classes_per_batch = super_classes_per_batch

        # checks
        assert (
            self.batch_size % super_classes_per_batch == 0
        ), f"batch_size should be a multiple of {super_classes_per_batch}"
        self.sub_batch_len = self.batch_size // super_classes_per_batch

        if self.samples_per_class != "all":
            assert self.samples_per_class > 0
            assert (
                self.sub_batch_len % self.samples_per_class == 0
            ), "batch_size not a multiple of samples_per_class"

        all_super_labels = set(labels[:, outer_label])
        self.super_image_lists = {slb: defaultdict(list) for slb in all_super_labels}
        for idx, instance in enumerate(labels):
            slb, lb = instance[outer_label], instance[inner_label]
            self.super_image_lists[slb][lb].append(idx)

        self.super_pairs = list(
            itertools.combinations(all_super_labels, super_classes_per_batch)
        )
        self.reshuffle()

    def __iter__(
        self,
    ):
        self.reshuffle()
        for batch in self.batches:
            yield [int(x) for x in batch]

    def __len__(
        self,
    ):
        return len(self.batches)

    def reshuffle(self):
        batches = []
        for combinations in self.super_pairs:

            for b in range(self.batches_per_super_tuple):

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    np.random.shuffle(all_classes)
                    for cl in all_classes:
                        if len(sub_batch) >= self.sub_batch_len:
                            break
                        instances = self.super_image_lists[slb][cl]
                        samples_per_class = (
                            self.samples_per_class
                            if self.samples_per_class != "all"
                            else len(instances)
                        )
                        if len(sub_batch) + samples_per_class > self.sub_batch_len:
                            continue
                        sub_batch.extend(
                            safe_random_choice(instances, size=samples_per_class)
                        )

                    batch.extend(sub_batch)

                np.random.shuffle(batch)
                batches.append(batch)

        np.random.shuffle(batches)
        self.batches = batches
