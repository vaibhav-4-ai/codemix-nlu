"""Stage 4: train the real model - one shared reader with two output heads.

    sentence -> backbone (hing-mbert) -> one vector per word-piece
                 |- intent head: reads the [CLS] vector  -> which of 60 intents
                 |- slot head:   reads every piece's vector -> which tag (B-time, O, ...)

One training step, in plain words:
  1. optimizer.zero_grad()  - forget the nudges worked out for the previous batch
  2. model(...)             - guess intent + tags for this batch
  3. loss                   - how wrong the guesses were (intent loss + slot loss)
  4. loss.backward()        - work out which way to nudge every weight to be less wrong
  5. optimizer.step()       - nudge them a little
No GPU (e.g. on GitHub): a tiny model runs the SAME loop for 2 steps, just to prove it works.
"""

import json
import random
import time
from pathlib import Path

import torch
from seqeval.metrics import f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from cmx.examples import align_tags, test_sets, training_examples

BACKBONE = "l3cube-pune/hing-mbert"
TINY_BACKBONE = "google/bert_uncased_L-2_H-128_A-2"  # for the no-GPU check only
LEARNING_RATE = 3e-5
EPOCHS = 4
BATCH_SIZE = 32
SLOT_WEIGHT = 1.0  # loss = intent loss + SLOT_WEIGHT * slot loss
MAX_PIECES = 64


class JointModel(nn.Module):
    def __init__(self, backbone, n_intents, n_tags):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(size, n_intents)
        self.slot_head = nn.Linear(size, n_tags)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        intent_logits = self.intent_head(hidden[:, 0])  # position 0 is the [CLS] piece
        slot_logits = self.slot_head(hidden)
        return intent_logits, slot_logits


def encode(tokenizer, examples, intent_ids, tag_ids):
    """Turn each example into piece numbers + the answers the model should learn."""
    rows = []
    for example in examples:
        enc = tokenizer(
            example["words"], is_split_into_words=True, truncation=True, max_length=MAX_PIECES
        )
        word_ids = enc.word_ids()
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "word_ids": word_ids,
                "tag_labels": align_tags(word_ids, example["tags"], tag_ids),
                "intent": intent_ids.get(example["intent"], -100),
            }
        )
    return rows


def make_batch(rows, pad_id, device):
    """Pad every row to the longest one in the batch, so they stack into one table."""
    longest = 0
    for row in rows:
        longest = max(longest, len(row["input_ids"]))
    ids, mask, tags, intents = [], [], [], []
    for row in rows:
        extra = longest - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * extra)
        mask.append([1] * len(row["input_ids"]) + [0] * extra)
        tags.append(row["tag_labels"] + [-100] * extra)
        intents.append(row["intent"])
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": torch.tensor(mask, device=device),
        "tags": torch.tensor(tags, device=device),
        "intents": torch.tensor(intents, device=device),
    }


def predict(model, rows, examples, tokenizer, intent_names, tag_names, device):
    """Guess intent (with probabilities) and one tag per word for every example."""
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            batch = make_batch(batch_rows, tokenizer.pad_token_id, device)
            intent_logits, slot_logits = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(intent_logits.float(), dim=-1).tolist()
            slot_best = slot_logits.argmax(dim=-1).tolist()
            for k, row in enumerate(batch_rows):
                example = examples[start + k]
                word_tags = ["O"] * len(example["words"])
                previous = None
                for position, word_id in enumerate(row["word_ids"]):
                    if word_id is not None and word_id != previous:
                        word_tags[word_id] = tag_names[slot_best[k][position]]
                    previous = word_id
                best = probs[k].index(max(probs[k]))
                results.append(
                    {
                        "id": example["id"],
                        "intent": example["intent"],
                        "predicted": intent_names[best],
                        "probs": [round(p, 5) for p in probs[k]],
                        "tags": example["tags"],
                        "predicted_tags": word_tags,
                    }
                )
    model.train()
    return results


def score(results):
    """Intent accuracy and slot F1 (seqeval: a slot counts only if every word of it is right)."""
    right = 0
    true_tags, guessed_tags = [], []
    for r in results:
        if r["predicted"] == r["intent"]:
            right += 1
        true_tags.append(r["tags"])
        guessed_tags.append(r["predicted_tags"])
    return {
        "intent_accuracy": right / len(results),
        "slot_f1": f1_score(true_tags, guessed_tags),
        "n": len(results),
    }


def write_jsonl(path, rows):
    lines = []
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def train(recipe, seed, backbone):
    random.seed(seed)
    torch.manual_seed(seed)
    on_gpu = torch.cuda.is_available()
    device = "cuda" if on_gpu else "cpu"

    examples = training_examples(recipe)
    tests = test_sets()
    epochs = EPOCHS
    if not on_gpu:
        # No GPU: tiny model, 20 examples, 2 steps - only proves the loop runs end to end.
        backbone = TINY_BACKBONE
        examples = examples[:20]
        small = {}
        for name, items in tests.items():
            small[name] = items[:20]
        tests = small
        epochs = 1
    batch_size = BATCH_SIZE if on_gpu else 10

    intent_names = []
    tag_names = ["O"]
    for example in examples:
        if example["intent"] not in intent_names:
            intent_names.append(example["intent"])
        for tag in example["tags"]:
            if tag not in tag_names:
                tag_names.append(tag)
    intent_names.sort()
    intent_ids = {}
    for number, name in enumerate(intent_names):
        intent_ids[name] = number
    tag_ids = {}
    for number, name in enumerate(tag_names):
        tag_ids[name] = number

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_rows = encode(tokenizer, examples, intent_ids, tag_ids)
    test_rows = {}
    for name, items in tests.items():
        test_rows[name] = encode(tokenizer, items, intent_ids, tag_ids)

    model = JointModel(backbone, len(intent_names), len(tag_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = epochs * ((len(train_rows) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()  # skips every label that is -100
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)  # mixed precision: faster on a T4

    out = Path("artifacts") / "train" / f"{recipe}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    curve_lines = ["epoch,train_loss,dev_intent_accuracy,dev_slot_f1,seconds\n"]
    print(f"backbone {backbone} | {len(train_rows):,} training sentences | device {device}")

    for epoch in range(1, epochs + 1):
        started = time.time()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(order), batch_size):
            batch_rows = []
            for i in order[start : start + batch_size]:
                batch_rows.append(train_rows[i])
            batch = make_batch(batch_rows, tokenizer.pad_token_id, device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=on_gpu):
                intent_logits, slot_logits = model(batch["input_ids"], batch["attention_mask"])
                intent_loss = loss_fn(intent_logits, batch["intents"])
                slot_loss = loss_fn(slot_logits.reshape(-1, len(tag_names)), batch["tags"].reshape(-1))
                loss = intent_loss + SLOT_WEIGHT * slot_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item()
            steps += 1
            if not on_gpu and steps == 2:
                break

        dev = score(predict(model, test_rows["dev"], tests["dev"], tokenizer, intent_names, tag_names, device))
        seconds = round(time.time() - started)
        print(
            f"epoch {epoch}: train loss {loss_sum / steps:.3f} | dev intent {dev['intent_accuracy']:.1%}"
            f" | dev slot F1 {dev['slot_f1']:.1%} | {seconds}s"
        )
        curve_lines.append(
            f"{epoch},{loss_sum / steps:.4f},{dev['intent_accuracy']:.4f},{dev['slot_f1']:.4f},{seconds}\n"
        )
    (out / "curves.csv").write_text("".join(curve_lines), encoding="utf-8")

    metrics = {"recipe": recipe, "seed": seed, "backbone": backbone, "on_gpu": on_gpu}
    for name in ["test", "devanagari_50", "hinglish_50"]:
        results = predict(model, test_rows[name], tests[name], tokenizer, intent_names, tag_names, device)
        write_jsonl(out / f"predictions_{name}.jsonl", results)
        metrics[name] = score(results)
        print(
            f"{name:14} intent {metrics[name]['intent_accuracy']:.1%}"
            f" | slot F1 {metrics[name]['slot_f1']:.1%} | n={metrics[name]['n']}"
        )
    metrics["intent_names"] = intent_names
    metrics["tag_names"] = tag_names
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if on_gpu:
        torch.save(model.state_dict(), out / "model.pt")
        tokenizer.save_pretrained(out / "tokenizer")
    print(f"saved everything in {out}/")
