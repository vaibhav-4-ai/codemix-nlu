import math

from cmx.calibrate import (
    answered_share,
    confidence,
    ece,
    fit_temperature,
    pick_cutoff,
    reliability,
)


def test_ece_perfect_and_overconfident():
    # 90% sure, right 9 times out of 10 -> honest
    assert ece([0.9] * 10, [True] * 9 + [False]) < 1e-9
    # 90% sure, right half the time -> 0.4 too confident
    assert abs(ece([0.9] * 10, [True] * 5 + [False] * 5) - 0.4) < 1e-9


def test_temperature_softens_confidence():
    case = {"logs": [math.log(0.8), math.log(0.2)], "answer": 0, "right": True}
    assert abs(confidence(case, 1.0) - 0.8) < 1e-9
    assert confidence(case, 2.0) < 0.8


def test_fit_temperature_softens_an_overconfident_model():
    # always 99% sure, but right only 60% of the time -> T should be above 1
    cases = []
    for i in range(10):
        cases.append({"logs": [math.log(0.99), math.log(0.01)], "answer": 0 if i < 6 else 1, "right": i < 6})
    assert fit_temperature(cases) > 1.0


def test_cutoff_keeps_answered_sentences_right():
    # low-confidence answers are the wrong ones -> the cut-off should drop them
    confidences = [0.95, 0.9, 0.85, 0.8, 0.3, 0.2]
    correct = [True, True, True, True, False, False]
    cutoff = pick_cutoff(confidences, correct)
    share, accuracy = answered_share(confidences, correct, cutoff)
    assert 0.2 < cutoff <= 0.8
    assert accuracy == 1.0
    assert abs(share - 4 / 6) < 1e-9


def test_reliability_points():
    points = reliability([0.95, 0.95, 0.15], [True, False, False])
    assert points == [(0.15, 0.0, 1), (0.95, 0.5, 2)]
