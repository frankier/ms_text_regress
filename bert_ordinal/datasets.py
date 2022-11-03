"""
This module contains functionality to load and preprocess open data ordinal
regression datasets.
"""

from typing import List, Tuple, Union

import datasets


def dec_label(example):
    return {"label": example["label"] - 1}


def get_dataset_scale_points_cdr(dataset) -> List[int]:
    # XXX: Can we do this without a full dataset scan?
    dataset_scale_points = [0] * dataset["train"].features["dataset"].num_classes

    def process_row(row):
        dataset_scale_points[row["dataset"]] = row["scale_points"]

    dataset.map(process_row)
    return dataset_scale_points


def get_dataset_scale_points_rt(dataset) -> List[int]:
    dataset_scale_points = []

    def process_row(row):
        if len(dataset_scale_points) < row["task_ids"] + 1:
            dataset_scale_points.extend(
                [0] * (row["task_ids"] - len(dataset_scale_points) + 1)
            )
        dataset_scale_points[row["task_ids"]] = row["scale_points"]

    dataset.map(process_row)
    return dataset_scale_points


def load_data(name: str) -> Tuple[datasets.DatasetDict, Union[int, List[int]], bool]:
    """
    Loads either the single part/single task dataset "shoe_reviews" or the multi
    part/multi task "cross_domain_reviews" dataset.
    """
    dataset: datasets.DatasetDict
    num_labels: Union[int, List[int]]
    is_multi = False
    if name == "shoe_reviews":
        d = datasets.load_dataset("juliensimon/amazon-shoe-reviews")
        assert isinstance(d, datasets.DatasetDict)
        dataset = d
        dataset = dataset.rename_column("labels", "label")
        num_labels = 5
    elif name == "cross_domain_reviews":
        d = datasets.load_dataset("frankier/cross_domain_reviews")
        assert isinstance(d, datasets.DatasetDict)
        dataset = d
        dataset = dataset.rename_column("rating", "label")
        dataset = dataset.map(dec_label)
        num_labels = get_dataset_scale_points_cdr(dataset)
        dataset = dataset.rename_column("dataset", "task_ids")
        is_multi = True
    elif name == "multiscale_rt_critics":
        d = datasets.load_dataset("frankier/processed_multiscale_rt_critics")
        assert isinstance(d, datasets.DatasetDict)
        # DatasetDict({k: v for k, v in d.items() if k in ("train", "test")})
        dataset = d
        dataset = dataset.rename_column("num", "label")
        dataset = dataset.rename_column("denom", "scale_points")
        dataset = dataset.map(dec_label)
        dataset = dataset.rename_column("group_id", "task_ids")
        dataset = dataset.rename_column("review_content", "text")
        num_labels = get_dataset_scale_points_rt(dataset)
        is_multi = True
    else:
        raise RuntimeError("Unknown dataset")
    return dataset, num_labels, is_multi
