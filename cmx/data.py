"""Download MASSIVE, keep the Hindi + English files, and pick sentences for you to rewrite."""

import hashlib
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from cmx.massive import slots

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SIZE = 40_251_390  # bytes, checked 2026-11-11
SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"  # measured 2026-11-18
KEEP = ("id", "partition", "scenario", "intent", "utt", "annot_utt")
DATA = Path("data")
SEED = 13


def load(locale: str) -> list[dict]:
    """Read one saved language file, e.g. load('hi-IN')."""
    path = DATA / "massive" / f"{locale}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def pick_seed(hindi: list[dict], english: list[dict], n: int = 50) -> list[dict]:
    """Pick n test sentences, each from a different intent, so the 50 cover many kinds of request."""
    english_by_id = {r["id"]: r for r in english}
    by_intent: dict[str, list[dict]] = {}
    for r in hindi:
        if r["partition"] == "test" and r["id"] in english_by_id and slots(r["annot_utt"]):
            by_intent.setdefault(r["intent"], []).append(r)
    rng = random.Random(SEED)
    intents = sorted(by_intent)
    rng.shuffle(intents)
    picked = []
    for intent in intents[:n]:
        r = rng.choice(by_intent[intent])
        picked.append(
            {
                "id": r["id"],
                "intent": r["intent"],
                "hindi": r["annot_utt"],
                "english": english_by_id[r["id"]]["annot_utt"],
            }
        )
    return picked


def pick_per_intent(rows, partition, per_intent, first_ids, seed):
    """Up to `per_intent` sentences of each intent from one partition.

    Sentences in `first_ids` go first (so the 50 already-done ones stay in the test set),
    then sentences that have slots, then the rest. Random order inside each group, fixed seed.
    """
    rng = random.Random(seed)
    by_intent = {}
    for r in rows:
        if r["partition"] == partition:
            if r["intent"] not in by_intent:
                by_intent[r["intent"]] = []
            by_intent[r["intent"]].append(r)
    picked = []
    for intent in sorted(by_intent):
        group = by_intent[intent]
        rng.shuffle(group)
        already = []
        with_slots = []
        without = []
        for r in group:
            if r["id"] in first_ids:
                already.append(r)
            elif slots(r["annot_utt"]):
                with_slots.append(r)
            else:
                without.append(r)
        picked += (already + with_slots + without)[:per_intent]
    return picked


def pick_sets() -> None:
    """Write the full test set (up to 10 per intent) and dev set (up to 7 per intent) to rewrite."""
    hindi = load("hi-IN")
    english_by_id = {}
    for r in load("en-US"):
        english_by_id[r["id"]] = r
    seed_ids = set()
    for line in (DATA / "annotation" / "seed_50.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            seed_ids.add(json.loads(line)["id"])
    for name, partition, per_intent in [("test_set", "test", 10), ("dev_set", "dev", 7)]:
        rows = []
        for r in pick_per_intent(hindi, partition, per_intent, seed_ids, SEED):
            rows.append(
                {
                    "id": r["id"],
                    "intent": r["intent"],
                    "hindi": r["annot_utt"],
                    "english": english_by_id[r["id"]]["annot_utt"],
                }
            )
        save_jsonl(DATA / "annotation" / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} sentences")


def fetch() -> None:
    """Download the tarball, check it, save hi-IN and en-US, and write the 50-sentence seed batch."""
    with urllib.request.urlopen(URL) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    print(f"downloaded {len(blob):,} bytes, sha256 {digest}")
    if len(blob) != SIZE or (SHA256 and digest != SHA256):
        raise SystemExit("the download does not match the expected file")

    out = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for locale in ("hi-IN", "en-US"):
            member = next(m for m in tar.getmembers() if m.name.endswith(f"/{locale}.jsonl"))
            rows = [json.loads(line) for line in tar.extractfile(member)]
            rows = [{k: r[k] for k in KEEP} for r in rows]
            save_jsonl(DATA / "massive" / f"{locale}.jsonl", rows)
            out[locale] = rows
            counts = {p: sum(r["partition"] == p for r in rows) for p in ("train", "dev", "test")}
            print(f"{locale}: {counts}")

    print("\nexample (same id in both languages):")
    first = out["hi-IN"][0]
    english = next(r for r in out["en-US"] if r["id"] == first["id"])
    print(f"  intent : {first['intent']}")
    print(f"  hindi  : {first['annot_utt']}")
    print(f"  english: {english['annot_utt']}")

    seed = pick_seed(out["hi-IN"], out["en-US"])
    save_jsonl(DATA / "annotation" / "seed_50.jsonl", seed)
    print(f"\nwrote {len(seed)} sentences for you to rewrite in data/annotation/seed_50.jsonl")
