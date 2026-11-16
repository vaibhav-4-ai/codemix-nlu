from cmx.quick import bootstrap_ci, romanize


def test_romanize_gives_english_letters():
    out = romanize("कमरे की बत्ती")
    assert out.isascii()
    assert out.startswith("kam")


def test_bootstrap_ci_contains_the_accuracy():
    correct = [True] * 30 + [False] * 20  # 60% accuracy
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.6 < hi
    assert 0 <= lo and hi <= 1
