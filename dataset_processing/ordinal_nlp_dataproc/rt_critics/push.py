import os
import sys

import datasets
from ordinal_nlp_dataproc.rt_critics.subdatasets import CONFIGS


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


def push_subdatasets():
    for config in CONFIGS:
        print("Pushing subdataset", config)
        push_dataset("subdatasets.py", "frankier/multiscale_rt_critics_subsets", config)


def main():
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            dataset = sys.argv[1]
            if dataset == "processed_multiscale_rt_critics":
                push_dataset("dataset.py", "frankier/processed_multiscale_rt_critics")
                return
            elif dataset == "multiscale_rt_critics_subsets":
                push_subdatasets()
                return
        push_dataset(
            "subdatasets.py", "frankier/multiscale_rt_critics_subsets", sys.argv[1]
        )
        return
    push_dataset("dataset.py", "frankier/processed_multiscale_rt_critics")
    push_subdatasets()


if __name__ == "__main__":
    main()
