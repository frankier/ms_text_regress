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
This dataset is a quick-and-dirty benchmark for predicting ratings across
different domains and on different rating scales based on text. It pulls in a
bunch of rating datasets, takes at most 1000 instances from each and combines
them into a big dataset.
"""

import operator
import os
import shutil
import sys
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Any, Callable

import datasets
from datasets import concatenate_datasets, load_dataset


@dataclass
class SubDataset:
    source: Any
    nick: str
    scale_points: int
    get_review: Callable[[Any], Any]
    get_rating: Callable[[Any], Any]


def warn(msg):
    print(file=sys.stderr)
    print(f" ** Warning: {msg} **", file=sys.stderr)
    print(file=sys.stderr)


@dataclass
class SplitHFSrc:
    name: str

    def load(self):
        return load_dataset(self.name, streaming=True)


@dataclass
class TrainOnlyHFSrc:
    name: str

    def load(self):
        if hasattr(self, "_cached"):
            return self._cached
        self._cached = load_dataset(self.name)["train"].train_test_split(
            test_size=0.5, seed=42
        )
        return self._cached


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
            dataset = load_dataset("csv", data_files=file_path + ".zip")
            return dataset["train"].train_test_split(test_size=0.5, seed=42)
        finally:
            shutil.rmtree(dir_name)


def int_or_drop(col):
    def inner(row):
        try:
            n = float(row[col])
        except ValueError:
            return None
        return round_near(n)

    return inner


gi = operator.itemgetter


def round_near(x, eps=0.001):
    x_rnd = int(x + 0.5)
    if abs(x_rnd - x) > eps:
        warn("got {x_rnd} when rounding {x}")
    return x_rnd


def dec(inner):
    def wrap(x):
        y = inner(x)
        if y is None:
            return None
        res = y - 1
        if res < 0:
            warn(
                "tried to convert 1-based index to 0-based index "
                "but ended up with negative"
            )
        return res

    return wrap


DATASETS = [
    SubDataset(
        SplitHFSrc("juliensimon/amazon-shoe-reviews"),
        "amazon-shoes",
        5,
        gi("text"),
        gi("labels"),
    ),
    # TODO: Appears to be corrupt
    # SubDataset("florentgbelidji/edmunds-car-ratings", "car-ratings", 40, lambda row: row["Review"].strip(), lambda row: round_near(row["Rating"] * 8) - 7),
    SubDataset(
        TrainOnlyHFSrc("florentgbelidji/car-reviews"),
        "car-ratings",
        5,
        gi("Review"),
        dec(gi("Rating")),
    ),
    SubDataset(
        SplitHFSrc("codyburker/yelp_review_sampled"),
        "yelp",
        5,
        gi("text"),
        dec(gi("stars")),
    ),
    SubDataset(
        SplitHFSrc("kkotkar1/course-reviews"),
        "course-reviews",
        5,
        gi("review"),
        dec(gi("label")),
    ),
    SubDataset(
        TrainOnlyHFSrc("app_reviews"), "app-reviews", 5, gi("review"), dec(gi("star"))
    ),
    SubDataset(
        TrainOnlyHFSrc("LoganKells/amazon_product_reviews_video_games"),
        "amazon-games",
        5,
        gi("reviewText"),
        lambda row: round_near(row["overall"]),
    ),
    SubDataset(
        KaggleSrc("zynicide/wine-reviews", "winemag-data-130k-v2.csv"),
        "wine-reviews",
        100,
        gi("description"),
        dec(gi("points")),
    ),
    SubDataset(
        KaggleSrc("sadmadlad/imdb-user-reviews", "Pulp Fiction/movieReviews.csv"),
        "imdb-user-reviews",
        10,
        gi("Review"),
        dec(int_or_drop("User's Rating out of 10")),
    ),
    # TODO: Unicode decoding error
    # SubDataset(KaggleSrc("arushchillar/disneyland-reviews", "DisneylandReviews.csv"), "disneyland-reviews", 5, gi("Review_Text"), gi("Rating")),
]

_DESCRIPTION = __doc__

_HOMEPAGE = ""

_LICENSE = "Mixed"


class CrossDomainReviews(datasets.GeneratorBasedBuilder):
    _DESCRIPTION

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "rating": datasets.Value("uint8"),
                "scale_points": datasets.Value("uint8"),
                "dataset": datasets.ClassLabel(names=[ds.nick for ds in DATASETS]),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            supervised_keys=("text", "rating"),
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
        key = 0
        for ds_info in DATASETS:
            subdataset = ds_info.source.load()
            lowest = float("inf")
            highest = float("-inf")
            got = 0
            for row in subdataset[split]:
                review = ds_info.get_review(row)
                if review is None:
                    continue
                rating = ds_info.get_rating(row)
                if rating is None:
                    continue
                assert (
                    0 <= rating < ds_info.scale_points
                ), f"Expected {rating} in half-open (Python-style) range [0, {ds_info.scale_points})"
                lowest = min(lowest, rating)
                highest = max(highest, rating)
                yield key, {
                    "text": review,
                    "rating": rating,
                    "scale_points": ds_info.scale_points,
                    "dataset": ds_info.nick,
                }
                key += 1
                got += 1
                if got >= 1000:
                    break
            if lowest != 0:
                warn(
                    f"Lowest rating in {ds_info.nick} was {lowest}, "
                    "would suppose it would be 0"
                )
            if highest != ds_info.scale_points - 1:
                warn(
                    f"Highest rating in {ds_info.nick} was {highest}, "
                    f"would suppose it would be {ds_info.scale_points - 1}"
                )
