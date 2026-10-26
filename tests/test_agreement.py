
import pytest

from xai_industrial.xai.agreement import jaccard


def test_jaccard_identical_sets():
    assert jaccard({"TP2", "H1"}, {"TP2", "H1"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"TP2"}, {"H1"}) == 0.0


def test_jaccard_partial_overlap():
    # Intersection = 2; union = 4.
    assert jaccard({"TP2", "H1", "TP3"}, {"TP2", "H1", "DV_pressure"}) == pytest.approx(0.5)


def test_jaccard_both_empty_sets():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_one_empty_set():
    assert jaccard({"TP2"}, set()) == 0.0


def test_jaccard_is_symmetric():
    a = {"TP2", "H1"}
    b = {"TP2", "DV_pressure"}
    assert jaccard(a, b) == jaccard(b, a)
