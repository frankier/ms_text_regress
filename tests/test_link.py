import torch
from pytest import approx
from torch.nn import functional as F

from bert_ordinal.element_link import get_link_by_name

LABELS = torch.tensor([0, 1, 2, 3, 4])
NUM_LABELS = 5
LABELS_ONE_HOT = F.one_hot(LABELS, num_classes=NUM_LABELS).float()


def assert_roundtrip(link, enc, test_preds=True):
    print(LABELS_ONE_HOT)
    if test_preds:
        print(link.label_dist_from_preds(enc))
        assert link.label_dist_from_preds(enc) == approx(LABELS_ONE_HOT)
        assert link.top_from_preds(enc) == approx(LABELS)
    print("enc")
    print(enc)
    assert all(link.top_from_logits(enc.logit(eps=1e-6)) == LABELS)


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
    assert_roundtrip(fwd_acat, enc, test_preds=False)


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
    assert_roundtrip(bwd_acat, enc, test_preds=False)
