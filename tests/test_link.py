import torch
from pytest import approx
from torch.nn import functional as F

from bert_ordinal.element_link import get_link_by_name

LABELS = torch.tensor([0, 1, 2, 3, 4])
NUM_LABELS = 5
LABELS_ONE_HOT = F.one_hot(LABELS, num_classes=NUM_LABELS).float()


def assert_roundtrip(link, enc):
    assert link.label_dist_from_preds(enc) == approx(LABELS_ONE_HOT)
    assert link.top_from_preds(enc) == approx(LABELS)
    assert all(link.top_from_logits(enc.logit(eps=1e-6)) == LABELS)


def assert_roundtrip_sim_logits(link, enc, weights):
    # Simulate some linear logits
    # Step 1: Make initial logits
    logits = enc.logit(eps=1e-6)
    # Step 2: Copy all 0 weighted logits from left/right neighbors
    for logits_vec, weights_vec in zip(logits.unbind(), weights.unbind()):
        left_val = None
        to_mask = []
        for i, (l, w) in enumerate(zip(logits_vec, weights_vec)):
            if w == 0.0:
                if left_val is not None:
                    logits_vec[i] = left_val
                else:
                    to_mask.append(i)
            else:
                if to_mask is not None:
                    for j in to_mask:
                        logits_vec[j] = l
                    to_mask = None
                left_val = l
    assert all(link.top_from_logits(logits) == LABELS)


def test_fwd_cumulative():
    fwd_cumulative = get_link_by_name("fwd_cumulative")
    enc, weight = fwd_cumulative.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert weight is None
    assert_roundtrip(fwd_cumulative, enc)


def test_bwd_cumulative():
    bwd_cumulative = get_link_by_name("bwd_cumulative")
    enc, weight = bwd_cumulative.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    )
    assert weight is None
    assert_roundtrip(bwd_cumulative, enc)


def test_fwd_sratio():
    fwd_sratio = get_link_by_name("fwd_sratio")
    enc, weight = fwd_sratio.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    )
    assert_roundtrip(fwd_sratio, enc)


def test_bwd_sratio():
    bwd_sratio = get_link_by_name("bwd_sratio")
    enc, weight = bwd_sratio.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert_roundtrip(bwd_sratio, enc)


def test_fwd_cratio():
    fwd_cratio = get_link_by_name("fwd_cratio")
    enc, weight = fwd_cratio.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    )
    assert_roundtrip(fwd_cratio, enc)


def test_bwd_cratio():
    bwd_cratio = get_link_by_name("bwd_cratio")
    enc, weight = bwd_cratio.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert_roundtrip(bwd_cratio, enc)


def test_fwd_acat():
    fwd_acat = get_link_by_name("fwd_acat")
    enc, weight = fwd_acat.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert_roundtrip_sim_logits(fwd_acat, enc, weight)


def test_bwd_acat():
    bwd_acat = get_link_by_name("bwd_acat")
    enc, weight = bwd_acat.link(LABELS, NUM_LABELS)
    assert enc == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert weight == approx(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert_roundtrip_sim_logits(bwd_acat, enc, weight)
