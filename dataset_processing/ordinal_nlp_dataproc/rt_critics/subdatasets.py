from typing import List

import pandas
from ordinal_nlp_dataproc.rt_critics.dataset import _DESCRIPTION as orig_description
from sklearn.model_selection import train_test_split

import datasets

_DESCRIPTION = __doc__ = (
    """
Subsampled versions of the rotten tomatoes critic reviews dataset. See original
description below:
"""
    + orig_description
)


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
    """
    This function downsamples large groups according to `group_keys` to the same
    size, while smaller groups are kept as-is, subject to producing a final
    sampling of size `max_dataset_size`. Groups with size less than
    `ignore_threshold` are either discarded or not taken into account in terms
    deciding which groups to downsample according to `discard_ignore`.
    """
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
    """
    This function downsamples large groups according to `outer_group_keys` to
    the approximately the same size, while smaller groups are kept as-is,
    subject to producing a final sampling of size `max_dataset_size`. The
    downsampling is done groupwise with respect to `inner_group_size`, so that
    these groups are not split up.
    """
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


def train_test_val_split(
    df, test_size=0.2, val_size=0.2, shuffle=True, random_state=None
):
    train_df, test_val_df = train_test_split(
        df, test_size=test_size + val_size, shuffle=shuffle, random_state=random_state
    )
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=test_size / (test_size + val_size),
        shuffle=shuffle,
        random_state=random_state,
    )
    return train_df, test_df, val_df


def _stratified_split_3way(
    split_groups, val_size=0.2, test_size=0.2, shuffle=True, random_state=None
):
    train_dfs = []
    test_dfs = []
    val_dfs = []
    for group_df in split_groups:
        train_df, test_df, val_df = train_test_val_split(
            group_df,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        val_dfs.append(val_df)
    return train_dfs, test_dfs, val_dfs


def stratified_split_3way(
    df, group_keys, val_size=0.2, test_size=0.2, shuffle=True, random_state=None
):
    train_dfs, val_dfs, test_dfs = _stratified_split_3way(
        (group_df for _key, group_df in df.groupby(group_keys)),
        val_size=val_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    return pandas.concat(train_dfs), pandas.concat(val_dfs), pandas.concat(test_dfs)


def _handle_small(df, group_keys, large_threshold, small_target, inner):
    if small_target == "separate":
        small_dfs = []
    split_groups = []
    small_groups = []
    for _key, group_df in df.groupby(group_keys):
        if len(group_df) < large_threshold:
            small_groups.append(group_df)
        else:
            split_groups.append(group_df)
    dfs = inner(split_groups)
    for group_df in small_groups:
        if small_target in dfs:
            dfs[small_target].append(group_df)
        elif small_target == "separate":
            small_dfs.append(group_df)
        elif small_target == "drop":
            pass
        else:
            raise ValueError(
                f"Unknown small_target {small_target}, must be one of train, test, separate, drop"
            )
    if small_target == "separate":
        dfs["small"] = pandas.concat(small_dfs)
    return tuple((pandas.concat(df) for df in dfs.values()))


def stratified_split_large(
    df,
    group_keys,
    large_threshold,
    test_size=0.25,
    small_target="train",
    shuffle=True,
    random_state=None,
):
    def inner(split_groups):
        train_dfs, test_dfs = _stratified_split(
            split_groups,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return {"train": train_dfs, "test": test_dfs}

    return _handle_small(df, group_keys, large_threshold, small_target, inner)


def stratified_split_large_3way(
    df,
    group_keys,
    large_threshold,
    val_size=0.2,
    test_size=0.2,
    small_target="train",
    shuffle=True,
    random_state=None,
):
    def inner(split_groups):
        train_dfs, test_dfs, val_dfs = _stratified_split_3way(
            split_groups,
            val_size=val_size,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return {"train": train_dfs, "test": test_dfs, "validation": val_dfs}

    return _handle_small(df, group_keys, large_threshold, small_target, inner)


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


def sample_multiscale_rt_critics(table):
    df = table.to_pandas()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = downsample_large_groups_groupwise(df, "grade_type", "group_id", 40000)
    df = renumber_groups(df, "group_id")
    return df


def sample_rt_critics_by_critic_min(df, min_reviews=500):
    df["orig_group_id"] = df["group_id"]
    # Drop groups with no critic_name given
    df = df[df["critic_name"].notna()]
    # Filter out critic names that are not natural names (e.g. E! staff)
    df = df[~df["critic_name"].str.lower().str.endswith("staff")]
    groups = df.groupby("critic_name")

    # Keep only biggest group from each critic
    def transform_group(grp):
        return grp[grp["group_id"] == grp.groupby("group_id").size().idxmax()]

    df = groups.apply(transform_group)
    groups = df.groupby(level="critic_name")
    df["group_id"] = groups.ngroup().astype(int)
    df = groups.filter(lambda x: len(x) >= min_reviews)
    df = renumber_groups(df, "group_id")
    df = df.reset_index(drop=True)
    return df.sample(frac=1, random_state=42)


def load_joined_rt_critics() -> datasets.Dataset:
    d = datasets.load_dataset("frankier/processed_multiscale_rt_critics")
    assert isinstance(d, datasets.DatasetDict)
    d = datasets.concatenate_datasets(list(d.values()))
    assert isinstance(d, datasets.Dataset)
    return d


def critics_by_critics_ds(min_reviews=500):
    d = load_joined_rt_critics()
    df = sample_rt_critics_by_critic_min(d.to_pandas(), min_reviews=min_reviews)
    train_df, test_df, val_df = stratified_split_3way(df, ["group_id"], random_state=42)
    dataset = pandas_dataset_dict(train_df, test_df, val_df)
    dataset = dataset.rename_columns(
        {
            "group_id": "task_ids",
            "review_content": "text",
        }
    )
    num_labels = get_dataset_scale_points_rt(dataset)
    is_multi = True
    return dataset, num_labels, is_multi


def pandas_dataset_dict(train, test, validation=None):
    d = {
        "train": datasets.Dataset.from_pandas(train, preserve_index=False),
        "test": datasets.Dataset.from_pandas(test, preserve_index=False),
    }
    if validation is not None:
        d["validation"] = datasets.Dataset.from_pandas(validation, preserve_index=False)
    return datasets.DatasetDict(d)


def load_dataset(name):
    if name == "rt_critics_one":
        d = load_joined_rt_critics()
        df = pandas.DataFrame(d)
        biggest_group_size = 0
        biggest_group_df = None
        for _key, group_df in df.groupby("group_id"):
            if len(group_df) > biggest_group_size:
                biggest_group_size = len(group_df)
                biggest_group_df = group_df
        train_df, test_df, val_df = train_test_val_split(
            biggest_group_df, val_size=0.2, test_size=0.2, shuffle=True, random_state=42
        )
        num_labels = int(train_df.iloc[0]["scale_points"])
        dataset = pandas_dataset_dict(train_df, test_df, val_df)
        dataset = dataset.rename_columns({"review_content": "text"})
        dataset = dataset.remove_columns("group_id")
        is_multi = False
    elif name == "rt_critics_by_critic_500pl":
        return critics_by_critics_ds(500)
    elif name == "rt_critics_by_critic_1000pl":
        return critics_by_critics_ds(1000)
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
        train_df, test_df, val_df = stratified_split_3way(
            df, ["group_id"], random_state=42
        )
        dataset = pandas_dataset_dict(train_df, test_df, val_df)
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
        df = sample_multiscale_rt_critics(d.data.table)
        train_df, test_df, val_df = stratified_split_large_3way(
            df, ["grade_type", "group_id"], 10, small_target="drop"
        )
        dataset = pandas_dataset_dict(train_df, test_df, val_df)
        dataset = dataset.rename_columns(
            {
                "group_id": "task_ids",
                "review_content": "text",
            }
        )
        num_labels = get_dataset_scale_points_rt(dataset)
        is_multi = True
    return dataset, num_labels, is_multi


COMMON_FEATURES = {
    "movie_title": datasets.Value("string"),
    "publisher_name": datasets.Value("string"),
    "critic_name": datasets.Value("string"),
    "text": datasets.Value("string"),
    "review_score": datasets.Value("string"),
    "grade_type": datasets.Value("string"),
    "orig_num": datasets.Value("float"),
    "orig_denom": datasets.Value("float"),
    "includes_zero": datasets.Value("bool"),
    "label": datasets.Value("uint8"),
    "scale_points": datasets.Value("uint8"),
    "multiplier": datasets.Value("uint8"),
}

_HOMEPAGE = ""

_LICENSE = "CC0"

_VERSION = datasets.Version("1.2.0")

CONFIGS = [
    "rt_critics_one",
    "rt_critics_by_critic_500pl",
    "rt_critics_by_critic_1000pl",
    "rt_critics_big_irregular_5",
    "multiscale_rt_critics",
]


class SubsampledMultiscaleRTCritics(datasets.GeneratorBasedBuilder):
    VERSION = _VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            version=_VERSION,
        )
        for name in CONFIGS
    ]

    def _info(self):
        if self.config.name == "rt_critics_one":
            features = datasets.Features(COMMON_FEATURES)
        elif self.config.name != "multiscale_rt_critics":
            features = datasets.Features(
                {
                    **COMMON_FEATURES,
                    **{
                        "task_ids": datasets.Value("uint32"),
                        "orig_group_id": datasets.Value("uint32"),
                    },
                }
            )
        else:
            features = datasets.Features(
                {**COMMON_FEATURES, **{"task_ids": datasets.Value("uint32")}}
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation="",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "validation"},
            ),
        ]

    def _generate_examples(self, split):
        if not hasattr(self, "_dataset"):
            self._dataset, self._num_labels, self._is_multi = load_dataset(
                self.config.name
            )
        for i, example in enumerate(self._dataset[split]):
            yield i, example
