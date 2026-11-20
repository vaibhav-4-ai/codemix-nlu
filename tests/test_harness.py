from cmx.harness import score


def make_item(i, intent, band):
    return {
        "id": str(i), "intent": intent, "band": band,
        "devanagari": {"words": ["a", "b"], "tags": ["B-time", "O"]},
        "hinglish": {"words": ["a", "b"], "tags": ["B-time", "O"]},
    }  # fmt: skip


def test_score_counts_the_drop_and_bands():
    items = [make_item(1, "alarm_set", "none"), make_item(2, "alarm_query", "heavy")]
    predictions = [
        {"id": 1, "form": "devanagari", "intent": "alarm_set", "tags": ["B-time", "O"]},
        {"id": 2, "form": "devanagari", "intent": "alarm_query", "tags": ["B-time", "O"]},
        {"id": 1, "form": "hinglish", "intent": "alarm_set", "tags": ["O", "O"]},
        {"id": 2, "form": "hinglish", "intent": "alarm_set", "tags": ["B-time", "O"]},
    ]
    result = score(items, predictions)
    assert result["devanagari"]["intent_accuracy"] == 1.0
    assert result["hinglish"]["intent_accuracy"] == 0.5
    assert result["drop"] == 0.5
    assert result["devanagari"]["slot_f1"] == 1.0
    # 2 true slots, 1 guessed, 1 right: F1 = 2 * 1 / (2 + 1)
    assert abs(result["hinglish"]["slot_f1"] - 2 / 3) < 1e-9
    assert result["hinglish_by_band"]["heavy"]["intent_accuracy"] == 0.0


def test_score_without_tags_gives_no_slot_f1():
    items = [make_item(1, "alarm_set", "none")]
    predictions = [
        {"id": "1", "form": "devanagari", "intent": "alarm_set"},
        {"id": "1", "form": "hinglish", "intent": "alarm_set"},
    ]
    result = score(items, predictions)
    assert result["hinglish"]["slot_f1"] is None
    assert result["drop"] == 0.0
