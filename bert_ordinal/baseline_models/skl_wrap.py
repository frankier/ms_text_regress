import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression


def fit(task_id, coefs, **kwargs):
    from scipy.stats import mode

    x, y = coefs
    model = LinearRegression(**kwargs)
    coefs = None
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            model = model.fit(x.reshape(-1, 1), y)
        except ValueError as exc:
            if not exc.args[0].startswith(
                "This solver needs samples of at least 2 classes"
            ):
                raise exc
            bias = mode(y, axis=None, keepdims=False).mode
            coefs = np.array([bias, 0])
        else:
            weight = model.coef_
            bias = model.intercept_
            coefs = np.array([bias, weight[0]])
    return task_id, coefs


def predict(
    coefs,
    task_ids,
    hiddens,
    batch_num_labels,
):

    from bert_ordinal.label_dist import clip_predictions_np

    preds = []
    for task_id, hidden, nl in zip(task_ids, hiddens, batch_num_labels, strict=True):
        if task_id not in coefs:
            preds.append(0)
            continue
        intercepts = coefs[task_id][0]
        coef = coefs[task_id][1]
        out = np.array(hidden.squeeze() * coef + intercepts)
        preds.append(clip_predictions_np(out, nl))
    return np.array(preds)
