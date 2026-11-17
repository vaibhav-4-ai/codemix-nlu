from cmx.examples import align_tags, make_example, romanized_copy


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


def test_align_tags_scores_only_the_first_piece_of_each_word():
    # pieces: [CLS] mu ##jhe ba ##je [SEP]  ->  words 0 0 1 1
    word_ids = [None, 0, 0, 1, 1, None]
    tag_ids = {"O": 0, "B-time": 1}
    labels = align_tags(word_ids, ["O", "B-time"], tag_ids)
    assert labels == [-100, 0, -100, 1, -100, -100]
