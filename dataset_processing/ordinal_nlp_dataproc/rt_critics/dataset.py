"""
Cleaned up version of the rotten tomatoes critic reviews dataset. The original
is obtained from Kaggle:
https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

Data has been scraped from the publicly available website
https://www.rottentomatoes.com as of 2020-10-31.

The clean up process drops anything without both a review and a rating, as well
as standardising the ratings onto several integer, ordinal scales.
"""

import os

import pandas
from sklearn.model_selection import train_test_split

import datasets


def split_dfs(df):
    train_dfs = []
    test_dfs = []
    split_groups = []
    small_groups = []
    for (publisher_name, grade_type), group_df in df.groupby(
        ["publisher_name", "grade_type"]
    ):
        if len(group_df) < 50:
            small_groups.append((publisher_name, grade_type, group_df))
        else:
            split_groups.append((publisher_name, grade_type, group_df))
    group_id = 0
    group_cols = {
        "publisher_name": [],
        "grade_type": [],
        "group_id": [],
        "scale_points": [],
    }

    def add_group(group_df, publisher_name, grade_type):
        nonlocal group_id
        group_cols["publisher_name"].append(publisher_name)
        group_cols["grade_type"].append(grade_type)
        group_cols["group_id"].append(group_id)
        group_cols["scale_points"].append(group_df.iloc[0]["scale_points"])
        group_id += 1

    for publisher_name, grade_type, group_df in split_groups:
        train_df, test_df = train_test_split(group_df, test_size=0.2)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        add_group(group_df, publisher_name, grade_type)
    for publisher_name, grade_type, group_df in small_groups:
        train_dfs.append(group_df)
        add_group(group_df, publisher_name, grade_type)
    train_df = pandas.concat(train_dfs)
    test_df = pandas.concat(test_dfs)
    group_id_df = pandas.DataFrame.from_dict(
        {k: v for k, v in group_cols.items() if k != "scale_points"}
    )
    group_id_df.set_index(["publisher_name", "grade_type"], inplace=True)
    train_df = train_df.join(group_id_df, on=["publisher_name", "grade_type"])
    test_df = test_df.join(group_id_df, on=["publisher_name", "grade_type"])
    df = df.join(group_id_df, on=["publisher_name", "grade_type"])
    group_df = pandas.DataFrame.from_dict(group_cols)
    return df, train_df, test_df, group_df


def get_datasets():
    movies_df = pandas.read_csv(os.environ["MOVIES_CSV"])
    review_df = pandas.read_parquet(os.environ["CRITIC_REVIEWS_PQ"])
    joined_df = review_df.join(
        movies_df.set_index("rotten_tomatoes_link"), "rotten_tomatoes_link"
    )
    all_df, train_df, test_df, group_df = split_dfs(joined_df)
    return (
        all_df,
        train_df,
        test_df,
        group_df,
    )


_DESCRIPTION = __doc__

_HOMEPAGE = ""

_LICENSE = "CC0"


def iter_pandas_df(df, cols):
    for tpl in df.itertuples():
        yield tpl.Index, {k: v for k, v in tpl._asdict().items() if k in cols}


NORMAL_FEATURES = datasets.Features(
    {
        "movie_title": datasets.Value("string"),
        "publisher_name": datasets.Value("string"),
        "critic_name": datasets.Value("string"),
        "review_content": datasets.Value("string"),
        "review_score": datasets.Value("string"),
        "grade_type": datasets.Value("string"),
        "orig_num": datasets.Value("float"),
        "orig_denom": datasets.Value("float"),
        "includes_zero": datasets.Value("bool"),
        "label": datasets.Value("uint8"),
        "scale_points": datasets.Value("uint8"),
        "multiplier": datasets.Value("uint8"),
        "group_id": datasets.Value("uint32"),
    }
)


class MultiscaleRTCritics(datasets.GeneratorBasedBuilder):
    _DESCRIPTION

    VERSION = datasets.Version("1.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=NORMAL_FEATURES,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
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
        ]

    def _generate_examples(self, split):
        if not hasattr(self, "_datasets"):
            self._datasets = get_datasets()
        all_dataset, train_dataset, test_dataset, group_df = self._datasets
        cols = set(NORMAL_FEATURES.keys())
        if split == "all":
            yield from iter_pandas_df(all_dataset, cols)
        elif split == "train":
            yield from iter_pandas_df(train_dataset, cols)
        elif split == "test":
            yield from iter_pandas_df(test_dataset, cols)
        # else:
        # yield from iter_pandas_df(group_df)
