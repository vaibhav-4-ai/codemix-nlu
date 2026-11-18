import random

from cmx.annotations import problems
from cmx.examples import (
    add_spelling_noise,
    align_tags,
    clean_word,
    make_example,
    romanized_copy,
)


def test_make_example_reads_the_brackets():
    example = make_example("1", "alarm_set", "mujhe [time : 8 baje] utha dena")
    assert example["words"] == ["mujhe", "8", "baje", "utha", "dena"]
    assert example["tags"] == ["O", "B-time", "I-time", "O", "O"]


def test_romanized_copy_keeps_the_tags():
    example = make_example("1", "alarm_set", "[time : आठ बजे] उठा")
    copy = romanized_copy(example)
    assert copy["tags"] == example["tags"]
    for word in copy["words"]:
        assert word.isascii()


def test_clean_word_drops_itrans_marks():
    assert clean_word("ha.nsii") == "hansii"
    assert clean_word("...") == "..."  # nothing left -> keep the original


def test_spelling_noise_is_seeded_and_ascii():
    first = add_spelling_noise("kamare", random.Random(1))
    second = add_spelling_noise("kamare", random.Random(1))
    assert first == second
    assert first.isascii()


def test_spelling_noise_can_drop_the_middle_a():
    seen = set()
    for seed in range(50):
        seen.add(add_spelling_noise("kamare", random.Random(seed)))
    assert "kamre" in seen


def test_align_tags_scores_only_the_first_piece_of_each_word():
    # pieces: [CLS] mu ##jhe ba ##je [SEP]  ->  words 0 0 1 1
    word_ids = [None, 0, 0, 1, 1, None]
    tag_ids = {"O": 0, "B-time": 1}
    labels = align_tags(word_ids, ["O", "B-time"], tag_ids)
    assert labels == [-100, 0, -100, 1, -100, -100]


def test_annotation_check_catches_a_missing_slot():
    hindi = "[date : कल] [time : सात बजे] उठा दो"
    assert problems(hindi, "[date : kal] [time : 7 baje] utha do") == ""
    assert "differ" in problems(hindi, "[date : kal] 7 baje utha do")
    assert problems(hindi, "[date : kal [time : 7 baje] utha do") != ""
