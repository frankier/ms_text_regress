import datasets

dataset = datasets.load_dataset("dataset.py")
dataset.push_to_hub("frankier/processed_multiscale_rt_critics")
