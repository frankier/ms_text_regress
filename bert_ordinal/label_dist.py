import torch

PRED_AVGS = ["median", "mode", "mean"]


def median_from_label_dist(label_dist):
    return (label_dist.cumsum(-1) > 0.5).nonzero()[0]


def mean_from_label_dist(label_dist):
    return sum(label_dist * torch.arange(len(label_dist)))


def mode_from_label_dist(label_dist):
    return label_dist.argmax()


def summarize_label_dist(label_dist):
    return {
        "median": median_from_label_dist(label_dist),
        "mode": mode_from_label_dist(label_dist),
        "mean": mean_from_label_dist(label_dist),
    }
