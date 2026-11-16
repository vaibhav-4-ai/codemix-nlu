import json

from cmx.quick import bootstrap_ci, export, make_model, predict_from_export, romanize


def test_romanize_gives_english_letters():
    out = romanize("कमरे की बत्ती")
    assert out.isascii()
    assert out.startswith("kam")


def test_bootstrap_ci_contains_the_accuracy():
    correct = [True] * 30 + [False] * 20  # 60% accuracy
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.6 < hi
    assert lo >= 0
    assert hi <= 1


def test_exported_model_gives_same_answers(tmp_path):
    # The demo page redoes the model's maths from the exported numbers.
    # If that maths drifts from scikit-learn's, the page would show wrong answers.
    x = ["light on karo", "light band karo", "alarm laga do", "alarm hata do",
         "gaana chalao", "song play karo", "mausam kaisa hai", "weather batao"] * 3
    y = ["light", "light", "alarm", "alarm", "music", "music", "weather", "weather"] * 3
    model = make_model().fit(x, y)
    export(model, tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    for text in ["kitchen ki light on karo", "subah ka alarm", "koi gaana", "aaj mausam"]:
        assert predict_from_export(data, text) == model.predict([text])[0]
