import numpy as np
import pytest
from sampling_lab import (
    LOGITS,
    entropy_bits,
    order_matters,
    sample_counts,
    softmax,
    support,
    top_k_filter,
    top_p_filter,
)

CONFIDENT = LOGITS["confident (capital of France is ...)"]
OPEN = LOGITS["open-ended (name a pet ...)"]


def test_probabilities_sum_to_one_for_every_setting():
    for logits in LOGITS.values():
        for t in (0.0, 0.2, 1.0, 3.0):
            p = softmax(logits, t)
            assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
            for top_p in (0.1, 0.5, 0.9, 1.0):
                assert top_p_filter(p, top_p).sum() == pytest.approx(1.0)


def test_temperature_one_is_plain_softmax_and_zero_is_greedy():
    expected = np.exp(OPEN) / np.exp(OPEN).sum()
    assert softmax(OPEN, 1.0) == pytest.approx(expected)
    greedy = softmax(OPEN, 0.0)
    assert greedy[0] == 1.0 and support(greedy) == 1


def test_higher_temperature_never_lowers_entropy_and_never_removes_tokens():
    for logits in LOGITS.values():
        entropies = [entropy_bits(softmax(logits, t)) for t in (0.1, 0.5, 1.0, 2.0, 5.0)]
        assert entropies == sorted(entropies)
        assert all(support(softmax(logits, t)) == len(logits) for t in (0.1, 1.0, 5.0))


def test_temperature_keeps_the_ranking_of_tokens():
    for logits in LOGITS.values():
        for t in (0.3, 1.0, 2.0):
            assert list(np.argsort(-softmax(logits, t), kind="stable")) == list(np.argsort(-logits, kind="stable"))


def test_top_p_keeps_the_smallest_prefix_that_reaches_p():
    rng = np.random.default_rng(0)
    for _ in range(200):
        probs = rng.dirichlet(np.ones(rng.integers(2, 15)))
        p = float(rng.uniform(0.05, 1.0))
        kept = top_p_filter(probs, p) > 0
        order = np.argsort(-probs, kind="stable")
        k = int(kept.sum())
        assert probs[order[:k]].sum() >= p - 1e-9  # reaches p
        assert k == 1 or probs[order[: k - 1]].sum() < p  # and one fewer would not
        assert kept[order[:k]].all()  # the kept ones are the most likely


def test_top_p_one_keeps_everything_and_tiny_p_keeps_only_the_top_token():
    p = softmax(OPEN, 1.0)
    assert support(top_p_filter(p, 1.0)) == len(p)
    assert support(top_p_filter(p, 0.001)) == 1


def test_top_p_adapts_to_the_distribution_but_top_k_does_not():
    kept = {name: support(top_p_filter(softmax(l, 1.0), 0.9)) for name, l in LOGITS.items()}
    assert kept["confident (capital of France is ...)"] == 1
    assert kept["open-ended (name a pet ...)"] >= 9
    assert {support(top_k_filter(softmax(l, 1.0), 5)) for l in LOGITS.values()} == {5}


def test_top_p_can_cut_the_tail_that_a_high_temperature_creates():
    hot = softmax(CONFIDENT, 2.0)
    assert support(hot) == len(CONFIDENT) and support(top_p_filter(hot, 0.9)) < len(CONFIDENT)


def test_the_order_of_temperature_and_top_p_changes_the_result():
    ambiguous = LOGITS["ambiguous (two good answers)"]
    a, b = order_matters(ambiguous, 2.0, 0.9)
    assert a.sum() == pytest.approx(1.0) and b.sum() == pytest.approx(1.0)
    assert support(a) > support(b) == 2


def test_sampled_frequencies_follow_the_probabilities():
    p = softmax(OPEN, 1.0)
    freq = sample_counts(p, 100_000, seed=3) / 100_000
    assert np.abs(freq - p).max() < 0.01


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        softmax(OPEN, -1.0)
    with pytest.raises(ValueError):
        top_p_filter(softmax(OPEN), 0.0)
    with pytest.raises(ValueError):
        top_p_filter(softmax(OPEN), 1.5)
    with pytest.raises(ValueError):
        top_k_filter(softmax(OPEN), 0)
