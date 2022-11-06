import os
import shutil
import sys
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Any, Callable

from datasets import load_dataset


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
