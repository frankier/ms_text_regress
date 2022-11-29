import torch

PRED_AVGS = ["median", "mode", "mean"]


def median_from_label_dist(label_dist):
    return (label_dist.cumsum(-1) <= 0.5).sum(-1)


def mean_from_label_dist(label_dist):
    return (
        label_dist * torch.arange(label_dist.size(-1), device=label_dist.device)
    ).sum(-1)


def mode_from_label_dist(label_dist):
    return label_dist.argmax(-1)


def summarize_label_dist(label_dist):
    return {
        "median": median_from_label_dist(label_dist),
        "mode": mode_from_label_dist(label_dist),
        "mean": mean_from_label_dist(label_dist),
    }
