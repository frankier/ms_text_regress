import torch

PRED_AVGS = ["median", "mode", "mean"]


def median_from_label_dist(label_dist):
    return (label_dist.cumsum(-1) <= 0.5).sum(-1)


def mean_from_label_dist(label_dist):
    return (
        label_dist * torch.arange(label_dist.size(-1), device=label_dist.device)
    ).sum(-1)


def rounded_mean_from_label_dist(label_dist):
    return (mean_from_label_dist(label_dist) + 0.5).int()


def mode_from_label_dist(label_dist):
    return label_dist.argmax(-1)


def summarize_label_dist(label_dist):
    return {
        "median": median_from_label_dist(label_dist),
        "mode": mode_from_label_dist(label_dist),
        "mean": rounded_mean_from_label_dist(label_dist),
    }


def summarize_label_dists(label_dists):
    res = {k: [] for k in PRED_AVGS}
    for label_dist in label_dists:
        label_dist_sum = summarize_label_dist(label_dist)
        res = {k: [*res[k], v] for k, v in label_dist_sum.items()}
    return {k: torch.hstack(v) for k, v in res.items()}


def clip_predictions_np(raw_predictions, batch_num_labels):
    import numpy as np

    return np.clip(raw_predictions.squeeze(-1) + 0.5, 0, batch_num_labels - 1).astype(
        int
    )
