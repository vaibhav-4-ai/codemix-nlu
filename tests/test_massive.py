from cmx.data import pick_seed
from cmx.massive import parse_annot, slot_types, slots


def test_parse_annot_tags_each_word():
    words, tags = parse_annot("kal [time : subah 7 baje] ka alarm laga do")
    assert words == ["kal", "subah", "7", "baje", "ka", "alarm", "laga", "do"]
    assert tags == ["O", "B-time", "I-time", "I-time", "O", "O", "O", "O"]


def test_parse_annot_works_on_devanagari():
    words, tags = parse_annot("[date : कल] [time : सात बजे] उठा देना")
    assert words == ["कल", "सात", "बजे", "उठा", "देना"]
    assert tags == ["B-date", "B-time", "I-time", "O", "O"]


def test_slot_helpers():
    annot = "[time : 7 baje] ka alarm [date : kal]"
    assert slot_types(annot) == ["date", "time"]
    assert slots(annot) == [("time", "7 baje"), ("date", "kal")]


def test_pick_seed_one_per_intent():
    hindi = [
        {"id": "1", "partition": "test", "intent": "alarm_set", "annot_utt": "[time : सात] उठा"},
        {"id": "2", "partition": "test", "intent": "alarm_set", "annot_utt": "[time : आठ] उठा"},
        {"id": "3", "partition": "test", "intent": "weather_query", "annot_utt": "[date : कल] मौसम"},
        {"id": "4", "partition": "train", "intent": "music_play", "annot_utt": "[song : x] चलाओ"},
        {"id": "5", "partition": "test", "intent": "general_joke", "annot_utt": "चुटकुला सुनाओ"},
    ]
    english = [{"id": i, "annot_utt": "en"} for i in "12345"]
    picked = pick_seed(hindi, english, n=50)
    # one per intent; train rows and slot-less rows are skipped
    assert sorted(p["intent"] for p in picked) == ["alarm_set", "weather_query"]
