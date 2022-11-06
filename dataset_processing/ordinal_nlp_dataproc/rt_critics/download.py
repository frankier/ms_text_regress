from dataclasses import dataclass
from os.path import join as pjoin

import click

KAGGLE_REPO = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"


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

    def load(self, dir_name):
        kaggle_api = get_kaggle_api()
        kaggle_api.dataset_download_file(self.name, self.file, path=dir_name)
        file_path = pjoin(dir_name, self.file.rsplit("/", 1)[-1])
        return file_path


@click.command()
@click.argument("output_path")
def main(output_path):
    for src in [
        KaggleSrc(KAGGLE_REPO, "rotten_tomatoes_movies.csv"),
        KaggleSrc(KAGGLE_REPO, "rotten_tomatoes_critic_reviews.csv"),
    ]:
        print("Downloaded", src.load(output_path))


if __name__ == "__main__":
    main()
