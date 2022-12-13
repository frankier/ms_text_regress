import torch
from torch.distributions.normal import Normal


def iter_task_normal_cutoffs(train_dataset):
    ndist = Normal(0, 1)
    df = train_dataset.to_pandas()
    groups = df.groupby("task_ids")
    for task_id, group in groups:
        print("** task_id", task_id, "**")
        labels = torch.tensor(group["label"].to_numpy(), dtype=torch.int)
        scale_points = group["scale_points"].iloc[0]
        counts = torch.bincount(labels, minlength=scale_points)
        smoothed_counts = counts.float() + 1.0 / scale_points
        print(smoothed_counts)
        freq = smoothed_counts / (len(labels) + 1)
        cutoffs = ndist.icdf(torch.clip(freq.cumsum(0), 0, 1))[:-1]
        yield task_id, cutoffs
