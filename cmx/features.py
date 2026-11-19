"""Code-mixing features: how much English is in a sentence, how often it switches, odd spellings.

Step 1 - give every word a language tag:
    "kal  subah  7  baje  alarm  set  kar  do"
     hi   hi     -  hi    en     en   hi   hi        (- = a number, no language)

Step 2 - count:
    CMI      = 100 * (1 - biggest language / tagged words) = 100 * (1 - 5/7) = 28.6
               0 = one language only, 50 = half and half (the most mixed two languages can be)
    I-Index  = language switches / (tagged words - 1) = 2 / 6 = 0.33   (baje->alarm, set->kar)
    script   = share of words written in English letters
    variant  = share of English-letter words that are a new SPELLING of a known Hindi word
               ("kamre" when training only saw "kamare")
    oov      = share of English-letter words that are in no word list at all

The word lists come from the training data only (never from a test sentence):
    English  every word of the English training sentences
    Hindi    every Hindi training word, turned into English letters ("kamare", "kii", ...)
"""

from cmx.data import load
from cmx.quick import romanize

# Short Hindi words people type that also exist in English ("me" = "in", "to" = "then").
# Counted as Hindi: in a Hinglish sentence they almost always are.
HINDI_SHORT_WORDS = [
    "me", "mein", "main", "hai", "h", "ko", "ki", "ka", "ke", "se", "pe", "par", "aur", "kya",
    "kar", "karo", "do", "to", "na", "nahi", "ho", "hu", "hoon", "tha", "thi", "ye", "yeh",
    "wo", "woh", "koi", "kuch", "bhi", "hi", "ek", "abhi", "bata", "batao", "kal", "aaj",
]  # fmt: skip

FEATURE_NAMES = ["cmi", "i_index", "english_share", "script_latin", "variant", "oov"]

BANDS = [
    ("none", 0.0, 0.001),  # CMI = 0: one language only
    ("light", 0.001, 20.0),
    ("medium", 20.0, 35.0),
    ("heavy", 35.0, 51.0),
]


def clean(word):
    """Lower case, letters and digits only: 'Alarm,' -> 'alarm'."""
    kept = ""
    for ch in word.lower():
        if ch.isalnum():
            kept += ch
    return kept


def is_devanagari(word):
    for ch in word:
        if "ऀ" <= ch <= "ॿ":
            return True
    return False


def skeleton(word):
    """A spelling-free key: first letter + the consonants after it.

    'kamare' and 'kamre' -> 'kmr';  'vaalaa' and 'wala' -> 'vl';  'nahii' and 'nahi' -> 'nh'.
    """
    word = word.replace("w", "v")
    if not word:
        return ""
    key = word[0]
    for ch in word[1:]:
        if ch not in "aeiouy" and ch != key[-1]:
            key += ch
    return key


class WordLists:
    """The English and Hindi word counts, built once from the training sentences."""

    def __init__(self, english_counts, hindi_counts):
        self.english = english_counts
        self.hindi = hindi_counts
        self.english_total = max(1, sum(english_counts.values()))
        self.hindi_total = max(1, sum(hindi_counts.values()))
        self.hindi_skeletons = set()
        for word in hindi_counts:
            self.hindi_skeletons.add(skeleton(word))

    def language(self, word):
        """'hi', 'en', or 'other' (numbers and punctuation)."""
        if is_devanagari(word):
            return "hi"
        word = clean(word)
        if word == "" or word.isdigit():
            return "other"
        if word in HINDI_SHORT_WORDS:
            return "hi"
        english_share = self.english.get(word, 0) / self.english_total
        hindi_share = self.hindi.get(word, 0) / self.hindi_total
        if english_share == 0 and hindi_share == 0:
            # Unknown word: Hindi spellings vary far more than English ones, so guess Hindi.
            return "hi"
        if english_share > hindi_share:
            return "en"
        return "hi"

    def spelling(self, word):
        """For an English-letter word: 'known', 'variant' or 'unknown'."""
        word = clean(word)
        if word in self.english or word in self.hindi or word in HINDI_SHORT_WORDS:
            return "known"
        if skeleton(word) in self.hindi_skeletons:
            return "variant"
        return "unknown"


def build_word_lists():
    english = {}
    for row in load("en-US"):
        if row["partition"] == "train":
            for word in row["utt"].split():
                key = clean(word)
                if key:
                    english[key] = english.get(key, 0) + 1
    hindi = {}
    for row in load("hi-IN"):
        if row["partition"] == "train":
            for word in row["utt"].split():
                if is_devanagari(word):
                    key = clean(romanize(word))
                    if key:
                        hindi[key] = hindi.get(key, 0) + 1
    return WordLists(english, hindi)


def sentence_features(words, lists):
    """All five numbers for one sentence (a list of words)."""
    tags = []
    latin = 0
    variant = 0
    unknown = 0
    for word in words:
        tag = lists.language(word)
        if tag != "other":
            tags.append(tag)
        if not is_devanagari(word) and clean(word) and not clean(word).isdigit():
            latin += 1
            kind = lists.spelling(word)
            if kind == "variant":
                variant += 1
            if kind == "unknown":
                unknown += 1

    n_hi = tags.count("hi")
    n_en = tags.count("en")
    n = n_hi + n_en
    cmi = 0.0
    if n > 0:
        cmi = 100 * (1 - max(n_hi, n_en) / n)
    switches = 0
    for i in range(1, len(tags)):
        if tags[i] != tags[i - 1]:
            switches += 1
    i_index = 0.0
    if n > 1:
        i_index = switches / (n - 1)

    letter_words = 0
    for word in words:
        if clean(word) and not clean(word).isdigit():
            letter_words += 1
    return {
        "cmi": round(cmi, 2),
        "i_index": round(i_index, 3),
        "english_share": round(n_en / n, 3) if n else 0.0,
        "script_latin": round(latin / letter_words, 3) if letter_words else 0.0,
        "variant": round(variant / latin, 3) if latin else 0.0,
        "oov": round(unknown / latin, 3) if latin else 0.0,
    }


def band(cmi):
    """'none', 'light', 'medium' or 'heavy' code-mixing."""
    for name, low, high in BANDS:
        if low <= cmi < high:
            return name
    return "heavy"
