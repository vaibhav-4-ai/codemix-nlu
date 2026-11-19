from cmx.evaluate import interval, macro_f1, slot_spans, summarize


def test_slot_spans_basic():
    assert slot_spans(["B-time", "I-time", "O", "B-date"]) == {(0, 1, "time"), (3, 3, "date")}


def test_slot_spans_i_without_b_starts_a_slot():
    assert slot_spans(["O", "I-place"]) == {(1, 1, "place")}


def test_slot_spans_type_change_splits():
    assert slot_spans(["B-time", "I-date"]) == {(0, 0, "time"), (1, 1, "date")}


def test_summarize_pools_seeds():
    # 2 sentences, 3 seeds: 3 + 0 right out of 6; 2 slots found out of 2 true + 2 guessed
    entries = [
        {"right": 3, "hits": 2, "true": 2, "guessed": 2},
        {"right": 0, "hits": 0, "true": 0, "guessed": 0},
    ]
    accuracy, slot_f1 = summarize(entries, 3)
    assert accuracy == 0.5
    assert slot_f1 == 1.0


def test_interval_middle_95():
    low, high = interval(list(range(1000)))
    assert low == 25
    assert high == 974


def test_macro_f1_weights_every_intent_equally():
    # alarm: 3 seeds right; music: 3 seeds wrong (guessed alarm)
    entries = [
        {"intent": "alarm", "guesses": ["alarm", "alarm", "alarm"]},
        {"intent": "music", "guesses": ["alarm", "alarm", "alarm"]},
    ]
    # alarm F1 = 2*3 / (3 + 6) = 0.667, music F1 = 0 -> average 0.333
    assert abs(macro_f1(entries) - (2 * 3 / 9) / 2) < 1e-9
