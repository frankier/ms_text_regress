"""
This module contains functionality to load and preprocess open data ordinal
regression datasets.
"""

import pickle
from os.path import join as pjoin
from typing import List, Optional, Tuple, Union

import datasets


def dec_label(row):
    return {"label": row["label"] - 1}


def get_dataset_scale_points_cdr(dataset) -> List[int]:
    # XXX: Can we do this without a full dataset scan?
    dataset_scale_points = [0] * dataset["train"].features["dataset"].num_classes

    def process_row(row):
        dataset_scale_points[row["dataset"]] = row["scale_points"]

    dataset.map(process_row, desc="Getting scale points")
    return dataset_scale_points


def get_dataset_scale_points_rt(dataset) -> List[int]:
    dataset_scale_points: List[int] = []

    def process_row(row):
        if len(dataset_scale_points) < row["task_ids"] + 1:
            dataset_scale_points.extend(
                [0] * (row["task_ids"] - len(dataset_scale_points) + 1)
            )
        dataset_scale_points[row["task_ids"]] = row["scale_points"]

    dataset.map(process_row, desc="Getting scale points")
    return dataset_scale_points


def load_data(
    name: str, num_dataset_proc: Optional[int] = None
) -> Tuple[datasets.DatasetDict, Union[int, List[int]], bool]:
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
        dataset = dataset.rename_columns({"rating": "label", "dataset": "task_ids"})
        dataset = dataset.map(
            dec_label, num_proc=num_dataset_proc, desc="Decrementing labels"
        )
        num_labels = get_dataset_scale_points_cdr(dataset)
        is_multi = True
    elif name in (
        "rt_critics_one",
        "rt_critics_by_critic_500pl",
        "rt_critics_by_critic_1000pl",
        "rt_critics_big_irregular_5",
        "multiscale_rt_critics",
    ):
        d = datasets.load_dataset("frankier/multiscale_rt_critics_subsets", name)
        assert isinstance(d, datasets.DatasetDict)
        dataset = d
        is_multi = name != "rt_critics_one"
        num_labels = get_dataset_scale_points_rt(dataset)
    else:
        raise RuntimeError("Unknown dataset")
    return dataset, num_labels, is_multi


def save_to_disk_with_labels(path, dataset, num_labels):
    dataset.save_to_disk(path)
    with open(pjoin(path, "num_labels.pkl"), "wb") as f:
        pickle.dump(num_labels, f)


def load_from_disk_with_labels(path):
    dataset = datasets.DatasetDict.load_from_disk(path, keep_in_memory=False)
    with open(pjoin(path, "num_labels.pkl"), "rb") as f:
        num_labels = pickle.load(f)
    return dataset, num_labels


def auto_dataset(dataset):
    try:
        return load_data(dataset)
    except RuntimeError:
        return load_from_disk_with_labels(dataset)
