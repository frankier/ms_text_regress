# Copyright 2022 Frankie Robertson and The HuggingFace Datasets Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cleaned up version of the rotten tomatoes critic reviews dataset. The original
is obtained from Kaggle:
https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

Data has been scraped from the publicly available website
https://www.rottentomatoes.com as of 2020-10-31.

The clean up process drops anything without both a review and a rating, as well
as standardising the ratings onto several integer, ordinal scales.
"""

import math
import operator
import os
import shutil
import sys
from dataclasses import dataclass
from fractions import Fraction
from os.path import join as pjoin
from typing import Any, Callable

import numpy
import pandas
from sklearn.model_selection import train_test_split

import datasets
from datasets import Dataset

KAGGLE_REPO = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
SHORT_LETTER_SCALE = ["F", "E", "D", "C", "B", "A"]
LONG_LETTER_SCALE = [
    "F-",
    "F",
    "F+" "E-",
    "E",
    "E+",
    "D-",
    "D",
    "D+",
    "C-",
    "C",
    "C+",
    "B-",
    "B",
    "B+",
    "A-",
    "A",
    "A+",
]


_kaggle_api = None


def get_kaggle_api():
    global _kaggle_api
    if _kaggle_api is not None:
        return _kaggle_api
    from kaggle.api.kaggle_api_extended import KaggleApi

    _kaggle_api = KaggleApi()
    _kaggle_api.authenticate()
    return _kaggle_api


@dataclass
class KaggleSrc:
    name: str
    file: str

    def load(self):
        if hasattr(self, "_cached"):
            return self._cached
        kaggle_api = get_kaggle_api()
        dir_name = self.name.replace("/", "__")
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        try:
            file_path = pjoin(dir_name, self.file.rsplit("/", 1)[-1])
            kaggle_api.dataset_download_file(self.name, self.file, path=dir_name)
            return pandas.read_csv(file_path + ".zip")
        finally:
            shutil.rmtree(dir_name)


def is_floatable(f):
    try:
        float(f)
        return True
    except ValueError:
        return False


def is_frac_str(s):
    bits = s.split("/")
    return len(bits) == 2 and is_floatable(bits[0]) and is_floatable(bits[1])


def is_barenum_str(s):
    return s.count("/") == 0 and is_floatable(s)


def is_dec_denom(s):
    bits = s.split("/")
    return len(bits) == 2 and "." in bits[1]


def drop_because(df, pred, reason):
    print(f"Dropping {pred.sum()} ({pred.mean() * 100:.2f}%) of reviews with {reason}")
    return df[~pred]


def drop_unrated(df):
    df = drop_because(df, df["review_score"].isna(), "no rating")
    df = drop_because(df, df["review_content"].isna(), "missing review")
    return df


def drop_odd_grade_types(df):
    is_any_letter = df["review_score"].isin(LONG_LETTER_SCALE)
    is_frac = df["review_score"].map(is_frac_str)
    is_barenum = df["review_score"].map(is_barenum_str)
    assert len(df[~is_frac & ~is_any_letter & ~is_barenum]) == 0
    df = drop_because(df, is_barenum, "bare number rating (i.e. no denominator)")
    is_frac_denom = df["review_score"].map(is_dec_denom)
    return drop_because(df, is_frac_denom, "fractional denominator")


def split_scores(df):
    nums = numpy.empty(len(df))
    denoms = numpy.empty(len(df))
    for idx, score in enumerate(df["review_score"]):
        if "/" in score:
            num, denom = score.split("/", 1)
            nums[idx] = float(num)
            denoms[idx] = float(denom)
        else:
            nums[idx] = nan
            denoms[idx] = nan
    df.insert(len(df.columns), "orig_num", nums)
    df.insert(len(df.columns), "orig_denom", denoms)


nan = float("nan")


def np_round(arr):
    return (arr + 0.5).astype(numpy.int32)


def process_letter_grade_group(group_df):
    group_df["includes_zero"] = False
    group_df["multiplier"] = 1
    group_df["non_neg_error"] = False
    if group_df.iloc[0]["letter_implies_short"]:
        group_df["label"] = SHORT_LETTER_SCALE.index(group_df.iloc[0]["review_score"])
        group_df["scale_points"] = len(SHORT_LETTER_SCALE)
    else:
        group_df["label"] = LONG_LETTER_SCALE.index(group_df.iloc[0]["review_score"])
        group_df["scale_points"] = len(LONG_LETTER_SCALE)
    return group_df


def process_includes_zero(group_df):
    multiplier = group_df.iloc[0]["multiplier"]
    includes_zero = any((label < multiplier for label in group_df["label"]))
    group_df["includes_zero"] = includes_zero
    if not includes_zero:
        group_df["label"] -= multiplier
        group_df["scale_points"] -= multiplier
    return group_df


def find_effective_nom_denom(group_df):
    if group_df.iloc[0]["is_any_letter"]:
        return process_letter_grade_group(group_df)
    else:
        group_df = common_denom_grades(group_df)
        return process_includes_zero(group_df)


def common_denom_grades(group_df):
    denoms = numpy.empty(len(group_df), dtype=numpy.int32)
    for idx, num in enumerate(group_df["orig_num"]):
        frac = Fraction.from_float(num)
        denoms[idx] = frac.limit_denominator(100).denominator
    common_denom = numpy.lcm.reduce(denoms)
    group_df["multiplier"] = common_denom
    num = common_denom * group_df["orig_num"].to_numpy()
    denom = common_denom * group_df["orig_denom"].to_numpy()
    group_df["label"] = np_round(num)
    group_df["scale_points"] = np_round(denom)
    group_df["non_neg_error"] = (abs(group_df["label"] - num) >= 0.05) | (
        abs(group_df["scale_points"] - denom) >= 0.05
    )
    return group_df


def normalize_reviews(review_df):
    print()
    # Drop unrated
    review_df = drop_unrated(review_df)

    # Strip whitespace from grades
    review_df["review_score"] = review_df["review_score"].str.replace(
        "\s+", "", regex=True
    )

    # Copy to get version to do calculations with
    working_review_df = review_df.copy()

    # Drop all rows where the review score occurs 2 or less times in the whole data set
    working_review_df = working_review_df.groupby("review_score").filter(
        lambda x: len(x) > 2
    )

    # Check/ensure that all grades are short letter, long letter, fraction or barenum
    working_review_df = drop_odd_grade_types(working_review_df)

    # Split fraction scores into numerator and denominator
    split_scores(working_review_df)

    # Divide letter scales into short and long
    # If a publisher has a mix of short and long, they're using long, otherwise short
    is_any_letter = working_review_df["review_score"].isin(LONG_LETTER_SCALE)
    is_short_letter = working_review_df["review_score"].isin(SHORT_LETTER_SCALE)
    # is_long_letter = is_any_letter & ~is_short_letter
    publisher_letter_implies_short = (
        pandas.DataFrame.from_dict(
            dict(
                publisher_name=working_review_df["publisher_name"],
                letter_implies_short=is_short_letter | ~is_any_letter,
            )
        )
        .groupby("publisher_name")
        .all()
    )
    working_review_df = working_review_df.join(
        publisher_letter_implies_short, on="publisher_name"
    )
    working_review_df["is_any_letter"] = is_any_letter

    # Now divide everything into grade types: either short letter, long letter
    # or the denominator of the fraction
    def get_grade_type(row):
        if row["is_any_letter"]:
            if row["letter_implies_short"]:
                return "short_letter"
            else:
                return "long_letter"
        else:
            return str(int(row["orig_denom"]))

    working_review_df["grade_type"] = working_review_df.apply(
        get_grade_type, axis="columns"
    )

    # Now we can filter out rare grade types
    working_review_df = working_review_df.join(
        working_review_df["grade_type"].value_counts().rename("grade_type_count"),
        on="grade_type",
    )
    working_review_df = drop_because(
        working_review_df,
        working_review_df["grade_type_count"] < 50,
        "grade type with less than 50 reviews",
    )

    # Print out some summary stats
    print("grades type counts")
    print(working_review_df["grade_type"].value_counts())
    print("unique grades", working_review_df["grade_type"].nunique())
    print("unique publishers", working_review_df["publisher_name"].nunique())
    print(
        "unique grade/publisher combinations",
        working_review_df.groupby(["grade_type", "publisher_name"]).ngroups,
    )

    # Now we can find common denominators on a (publisher, grade type) combination basis
    working_review_df = working_review_df.groupby(
        ["publisher_name", "grade_type"], group_keys=False
    ).apply(find_effective_nom_denom)
    working_review_df = drop_because(
        working_review_df, working_review_df["multiplier"] > 500, "multiplier > 500"
    )
    assert working_review_df["non_neg_error"].sum() == 0

    # More summary stats
    print("non-neg error count", working_review_df["non_neg_error"].sum())
    print("multipliers")
    print(working_review_df["multiplier"].value_counts())
    print("includes_zero")
    print(working_review_df["includes_zero"].value_counts())
    print("grade breakdown")
    print(
        working_review_df.value_counts(
            ["grade_type", "multiplier", "includes_zero", "scale_points"]
        )
    )

    # TODO: Add back in rare review_scores dropped at the beginning when they
    # are compatible with some common denominator + grade type from the same
    # publisher

    print("number of reviews left", len(working_review_df))
    print("reviews per publisher")
    print(working_review_df.value_counts(["publisher_name", "grade_type"]))

    # Delete working columns
    del working_review_df["letter_implies_short"]
    del working_review_df["is_any_letter"]
    del working_review_df["grade_type_count"]
    del working_review_df["non_neg_error"]

    return working_review_df


def save_normalised(output_path):
    review_df = KaggleSrc(KAGGLE_REPO, "rotten_tomatoes_critic_reviews.csv").load()
    review_df = normalize_reviews(review_df)
    review_df.to_csv(output_path)


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
    movies_df = KaggleSrc(KAGGLE_REPO, "rotten_tomatoes_movies.csv").load()
    review_df = KaggleSrc(KAGGLE_REPO, "rotten_tomatoes_critic_reviews.csv").load()
    review_df = normalize_reviews(review_df)
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
        "label": datasets.Value("uint8"),
        "scale_points": datasets.Value("uint8"),
        "multiplier": datasets.Value("uint8"),
        "group_id": datasets.Value("uint32"),
    }
)


class MultiscaleRTCritics(datasets.GeneratorBasedBuilder):
    _DESCRIPTION

    VERSION = datasets.Version("1.0.0")

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
