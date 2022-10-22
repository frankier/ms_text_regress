from pytest import approx


def test_qwk():
    from bert_ordinal.eval import qwk

    # Comparison values are from scikit-learn
    # >>> from sklearn.metrics import cohen_kappa_score
    # >>> cohen_kappa_score([0, 4, 3], [1, 4, 2], labels=[0, 1, 2, 3, 4], weights="quadratic")
    # 0.85
    assert qwk([0, 4, 3], [1, 4, 2], 5) == approx(0.85)
    # >>> cohen_kappa_score([0, 4, 4], [4, 0, 0], labels=[0, 1, 2, 3, 4], weights="quadratic")
    # -0.8000000000000003
    assert qwk([0, 4, 4], [4, 0, 0], 5) == approx(-0.8)
    # >>> cohen_kappa_score([0, 1], [2, 1], labels=[0, 1, 2], weights="quadratic")
    # -0.33333333333333326
    assert qwk([0, 1], [2, 1], 3) == approx(-1 / 3)


def test_qwk_multi_norm():
    from bert_ordinal.eval import qwk_multi_norm

    # These are just the same as test_qwk
    assert qwk_multi_norm([0, 4, 3], [1, 4, 2], [5, 5, 5]) == approx(0.85)
    assert qwk_multi_norm([0, 1], [2, 1], [3, 3]) == approx(-1 / 3)


def test_qwk_multi_norm_regression():
    from bert_ordinal.eval import qwk_multi_norm

    # XXX: This value was obtained by running the function. 0.4767441860465116
    # If it was linear it would be (0.85 * 3 + -1/3 * 2) / 5, but it's not
    assert qwk_multi_norm([0, 0, 4, 1, 3], [1, 2, 4, 1, 2], [5, 3, 5, 3, 5]) == approx(
        0.4767441860465116
    )
