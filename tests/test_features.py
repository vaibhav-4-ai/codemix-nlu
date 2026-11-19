from cmx.features import WordLists, band, sentence_features, skeleton


def tiny_lists():
    english = {"alarm": 10, "set": 8, "play": 6, "the": 20, "music": 5}
    hindi = {"kamare": 4, "kii": 6, "battii": 2, "subah": 3, "baje": 5}
    return WordLists(english, hindi)


def test_skeleton_ignores_spelling():
    assert skeleton("kamare") == skeleton("kamre") == "kmr"
    assert skeleton("vaalaa") == skeleton("wala") == "vl"


def test_cmi_for_a_mixed_sentence():
    # hi hi - hi en en hi hi : 5 Hindi, 2 English, 1 number
    words = ["kal", "subah", "7", "baje", "alarm", "set", "kar", "do"]
    values = sentence_features(words, tiny_lists())
    assert values["cmi"] == round(100 * (1 - 5 / 7), 2)
    assert values["i_index"] == round(2 / 6, 3)  # baje->alarm, set->kar


def test_one_language_has_cmi_zero():
    values = sentence_features(["play", "the", "music"], tiny_lists())
    assert values["cmi"] == 0.0
    assert band(values["cmi"]) == "none"


def test_new_spelling_counts_as_variant():
    lists = tiny_lists()
    assert lists.spelling("kamre") == "variant"
    assert lists.spelling("kamare") == "known"
    assert lists.spelling("xyzzy") == "unknown"


def test_bands():
    assert band(10) == "light"
    assert band(25) == "medium"
    assert band(50) == "heavy"
