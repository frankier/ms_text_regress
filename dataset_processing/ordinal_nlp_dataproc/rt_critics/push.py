import os

import datasets


def main():
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset.py"
    )
    dataset = datasets.load_dataset(dataset_path)
    dataset.push_to_hub("frankier/processed_multiscale_rt_critics")


if __name__ == "__main__":
    main()
