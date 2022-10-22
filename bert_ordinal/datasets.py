"""
This module contains functionality to load and preprocess open data ordinal
regression datasets.
"""

from typing import List, Tuple, Union

import datasets


def dec_label(example):
    return {"label": example["label"] - 1}


def get_dataset_scale_points(dataset):
    # XXX: Can we do this without a full dataset scan?
    dataset_scale_points = [0] * dataset["train"].features["dataset"].num_classes

    def process_row(row):
        dataset_scale_points[row["dataset"]] = row["scale_points"]

    dataset.map(process_row)
    return dataset_scale_points


def load_data(name: str) -> Tuple[datasets.Dataset, Union[int, List[int]], bool]:
    """
    Loads either the single part/single task dataset "shoe_reviews" or the multi
    part/multi task "cross_domain_reviews" dataset.
    """
    is_multi = False
    if name == "shoe_reviews":
        dataset = datasets.load_dataset("juliensimon/amazon-shoe-reviews")
        dataset = dataset.rename_column("labels", "label")
        num_labels = 5
    elif name == "cross_domain_reviews":
        dataset = datasets.load_dataset("frankier/cross_domain_reviews")
        dataset = dataset.rename_column("rating", "label")
        dataset = dataset.map(dec_label)
        num_labels = get_dataset_scale_points(dataset)
        dataset = dataset.rename_column("dataset", "task_ids")
        is_multi = True
    else:
        raise RuntimeError("Unknown dataset")
    return dataset, num_labels, is_multi
