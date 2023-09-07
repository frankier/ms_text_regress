import os
import sys

from ordinal_nlp_dataproc.rt_critics.subdatasets import CONFIGS

import datasets


def push_dataset(script, name, config=None):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if config is None:
        dataset = datasets.load_dataset(dataset_path)
    else:
        dataset = datasets.load_dataset(dataset_path, config)
    print("Pushing", name, config)
    if config is None:
        dataset.push_to_hub(name)
    else:
        dataset.push_to_hub(name, config_name=config)


def main():
    if len(sys.argv) > 1:
        push_dataset(
            "subdatasets.py", "frankier/multiscale_rt_critics_subsets", sys.argv[1]
        )
        return
    push_dataset("dataset.py", "frankier/processed_multiscale_rt_critics")
    for config in CONFIGS:
        push_dataset("subdatasets.py", "frankier/multiscale_rt_critics_subsets", config)


if __name__ == "__main__":
    main()
