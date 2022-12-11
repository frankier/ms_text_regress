"""
This module contains functionality to load and preprocess open data ordinal
regression datasets.
"""

import pickle
from os.path import join as pjoin
from typing import List, Optional, Tuple, Union

import pandas
import pyarrow
from sklearn.model_selection import train_test_split

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


def find_num_downsampled_groups(max_dataset_size, group_sizes):
    for num_downsampled_groups in range(1, len(group_sizes) + 1):
        next_group_size = (
            group_sizes.iloc[num_downsampled_groups]
            if num_downsampled_groups < len(group_sizes)
            else 0
        )
        rest_size = group_sizes.iloc[num_downsampled_groups:].sum()
        available_samples = max_dataset_size - rest_size
        samples_per_downsampled_group = available_samples / num_downsampled_groups
        if samples_per_downsampled_group >= next_group_size:
            break
    return num_downsampled_groups


def downsample_large_groups(
    df, group_keys, max_dataset_size, ignore_threshold=0, discard_ignore=False
):
    groups = df.groupby(group_keys)
    group_sizes = groups.size()
    group_sizes.sort_values(ascending=False, inplace=True)
    ignore_idx = (-group_sizes).searchsorted(-ignore_threshold, side="right")
    ignored_groups = group_sizes.iloc[ignore_idx:]
    group_sizes = group_sizes.iloc[:ignore_idx]
    num_downsampled_groups = find_num_downsampled_groups(max_dataset_size, group_sizes)
    # Add back in non-downsampled groups
    new_df_groups = [
        groups.get_group(group) for group in group_sizes.index[num_downsampled_groups:]
    ]
    if not discard_ignore:
        # Add back in ignored groups
        new_df_groups.extend(
            (groups.get_group(group) for group in ignored_groups.index)
        )
    for downsample_group_idx in range(num_downsampled_groups - 1, -1, -1):
        available_samples = max_dataset_size - sum(
            (len(group) for group in new_df_groups)
        )
        samples_per_downsampled_group = available_samples / (downsample_group_idx + 1)
        group = group_sizes.index[downsample_group_idx]
        new_df_groups.append(
            groups.get_group(group).iloc[: int(samples_per_downsampled_group + 0.5)]
        )
    return pandas.concat(new_df_groups)


def downsample_large_groups_groupwise(
    df, outer_group_keys, inner_group_keys, max_dataset_size
):
    groups = df.groupby(outer_group_keys)
    group_sizes = groups.size()
    group_sizes.sort_values(ascending=False, inplace=True)
    num_downsampled_groups = find_num_downsampled_groups(max_dataset_size, group_sizes)
    # Add back in non-downsampled groups
    new_df_groups = [
        groups.get_group(group) for group in group_sizes.index[num_downsampled_groups:]
    ]
    for downsample_group_idx in range(num_downsampled_groups):
        available_samples = max_dataset_size - sum(
            (len(group) for group in new_df_groups)
        )
        samples_per_downsampled_group = available_samples / (
            num_downsampled_groups - downsample_group_idx
        )
        group = group_sizes.index[downsample_group_idx]
        inner_groups = groups.get_group(group).groupby(inner_group_keys)
        inner_group_sizes = inner_groups.size()
        inner_group_sizes.sort_values(ascending=False, inplace=True)
        if downsample_group_idx < num_downsampled_groups - 1:
            side = "right"
        else:
            side = "left"
        sample_idx = inner_group_sizes.cumsum().searchsorted(
            samples_per_downsampled_group, side=side
        )
        new_df_groups.extend(
            (
                inner_groups.get_group(group)
                for group in inner_group_sizes.index[:sample_idx]
            )
        )
    return pandas.concat(new_df_groups)


def renumber_groups(df, group_key):
    groups = df.groupby(group_key)
    group_sizes = groups.size()
    group_sizes.sort_values(ascending=False, inplace=True)
    group_renumbering = {group: idx for idx, group in enumerate(group_sizes.index)}
    df[group_key] = df[group_key].map(group_renumbering)
    return df


def _stratified_split(split_groups, test_size=0.25, shuffle=True, random_state=None):
    train_dfs = []
    test_dfs = []
    for group_df in split_groups:
        train_df, test_df = train_test_split(
            group_df, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    return train_dfs, test_dfs


def stratified_split(df, group_keys, test_size=0.25, shuffle=True, random_state=None):
    train_dfs, test_dfs = _stratified_split(
        (group_df for _key, group_df in df.groupby(group_keys)),
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    return pandas.concat(train_dfs), pandas.concat(test_dfs)


def stratified_split_large(
    df,
    group_keys,
    large_threshold,
    test_size=0.25,
    small_target="train",
    shuffle=True,
    random_state=None,
):
    if small_target == "separate":
        small_dfs = []
    split_groups = []
    small_groups = []
    for _key, group_df in df.groupby(group_keys):
        if len(group_df) < large_threshold:
            small_groups.append(group_df)
        else:
            split_groups.append(group_df)
    train_dfs, test_dfs = _stratified_split(
        split_groups, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    for group_df in small_groups:
        if small_target == "train":
            train_dfs.append(group_df)
        elif small_target == "test":
            test_dfs.append(group_df)
        elif small_target == "separate":
            small_dfs.append(group_df)
        elif small_target == "drop":
            pass
        else:
            raise ValueError(
                f"Unknown small_target {small_target}, must be one of train, test, separate, drop"
            )
    train_df = pandas.concat(train_dfs)
    test_df = pandas.concat(test_dfs)
    if small_target == "separate":
        small_df = pandas.concat(small_dfs)
        return train_df, test_df, small_df
    else:
        return train_df, test_df


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


def sample_multiscale_rt_critics(batch):
    df = pandas.DataFrame.from_dict(batch.data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = downsample_large_groups_groupwise(df, "grade_type", "group_id", 40000)
    df = renumber_groups(df, "group_id")
    return pyarrow.Table.from_pandas(df)


def sample_rt_critics_by_critic_500pl(batch):
    df = pandas.DataFrame.from_dict(batch.data)
    df["orig_group_id"] = df["group_id"]
    # Drop groups with no critic_name given
    df = df[df["critic_name"].notna()]
    groups = df.groupby(["critic_name", "group_id"])
    df["group_id"] = groups.ngroup().astype(int)
    df = groups.filter(lambda x: len(x) >= 500)
    return pyarrow.Table.from_pandas(df.sample(frac=1, random_state=42))


def load_joined_rt_critics() -> datasets.Dataset:
    d = datasets.load_dataset("frankier/processed_multiscale_rt_critics")
    assert isinstance(d, datasets.DatasetDict)
    d = datasets.concatenate_datasets(list(d.values()))
    assert isinstance(d, datasets.Dataset)
    return d


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
    elif name == "rt_critics_one":
        d = load_joined_rt_critics()
        df = pandas.DataFrame(d)
        biggest_group_size = 0
        biggest_group_df = None
        for _key, group_df in df.groupby("group_id"):
            if len(group_df) > biggest_group_size:
                biggest_group_size = len(group_df)
                biggest_group_df = group_df
        train_df, test_df = train_test_split(
            biggest_group_df, test_size=0.25, shuffle=True, random_state=42
        )
        num_labels = int(train_df.iloc[0]["scale_points"])
        dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(train_df, preserve_index=False),
                "test": datasets.Dataset.from_pandas(test_df, preserve_index=False),
            }
        )
        dataset = dataset.rename_columns({"review_content": "text"})
        is_multi = False
    elif name == "rt_critics_by_critic_500pl":
        d = load_joined_rt_critics()
        d = d.map(sample_rt_critics_by_critic_500pl, batched=True, batch_size=None)
        df = pandas.DataFrame(d)
        train_df, test_df = stratified_split(df, ["group_id"], random_state=42)
        dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(train_df, preserve_index=False),
                "test": datasets.Dataset.from_pandas(test_df, preserve_index=False),
            }
        )
        dataset = dataset.rename_columns(
            {
                "group_id": "task_ids",
                "review_content": "text",
            }
        )
        num_labels = get_dataset_scale_points_rt(dataset)
        is_multi = True
    elif name == "rt_critics_big_irregular_5":
        d = load_joined_rt_critics()
        df = pandas.DataFrame(d)
        critic_publishers_to_select = [
            # Very few As. Prefers whole letter grades.
            ("Frank Swietek", "One Guy's Opinion"),
            ("Mark Dujsik", "Mark Reviews Movies"),
            ("Emanuel Levy", "EmanuelLevy.Com"),
            ("Ken Hanke", "Mountain Xpress (Asheville, NC)"),
            ("Chris Hewitt", "St. Paul Pioneer Press"),
        ]
        selected = []
        for critic_name, publisher in critic_publishers_to_select:
            sub_df = df[
                (df["critic_name"] == critic_name) & (df["publisher_name"] == publisher)
            ]
            assert len(sub_df) > 0
            # Pick the largest grade_type from each critic/publisher combo
            sub_df = sub_df[sub_df["grade_type"] == sub_df["grade_type"].mode()[0]]
            print("Selected", len(sub_df), "reviews from", critic_name, "at", publisher)
            selected.append(sub_df)
        df = pandas.concat(selected)
        groups = df.groupby(["critic_name", "group_id"])
        df["orig_group_id"] = df["group_id"]
        df["group_id"] = groups.ngroup().astype(int)
        train_df, test_df = stratified_split(df, ["group_id"], random_state=42)
        dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(train_df, preserve_index=False),
                "test": datasets.Dataset.from_pandas(test_df, preserve_index=False),
            }
        )
        dataset = dataset.rename_columns(
            {
                "group_id": "task_ids",
                "review_content": "text",
            }
        )
        num_labels = get_dataset_scale_points_rt(dataset)
        is_multi = True
    elif name == "multiscale_rt_critics":
        d = load_joined_rt_critics()
        d = d.map(sample_multiscale_rt_critics, batched=True, batch_size=None)
        """
        df = pandas.DataFrame(d["train"])
        df = df.sample(frac=1).reset_index(drop=True)
        df = downsample_large_groups_groupwise(df, "grade_type", "group_id", 40000)
        df = renumber_groups(df, "group_id")
        """
        # df = stratified_split(df, ["grade_type", "group_id"], shuffle=True)
        # train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['grade_type', 'group_id']].to_records(index=False))
        df = pandas.DataFrame(d)
        train_df, test_df = stratified_split_large(
            df, ["grade_type", "group_id"], 10, small_target="drop"
        )
        dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(train_df, preserve_index=False),
                "test": datasets.Dataset.from_pandas(test_df, preserve_index=False),
            }
        )
        dataset = dataset.rename_columns(
            {
                "group_id": "task_ids",
                "review_content": "text",
            }
        )
        num_labels = get_dataset_scale_points_rt(dataset)
        is_multi = True
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
