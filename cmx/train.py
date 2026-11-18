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
With grad_accum = 2, steps 2-4 run twice (two half-batches) before ONE step 5: same result as a
batch twice as big, with half the GPU memory.

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
MAX_PIECES = 64
BEST_PARAMS = Path("configs") / "best_params.json"

# Used until Optuna (cmx tune) has written configs/best_params.json.
DEFAULT_PARAMS = {
    "learning_rate": 3e-5,
    "epochs": 4,
    "batch_size": 32,
    "grad_accum": 1,
    "slot_weight": 1.0,  # loss = intent loss + slot_weight * slot loss
    "rare_slot_weighting": False,  # True = mistakes on rare slot tags cost more
    "warmup": 0.1,  # share of steps where the learning rate ramps up from 0
    "dropout": 0.1,
}
REFERENCE_RECIPE = "fixed"
SCORED_SETS = [
    "dev",
    "test",
    "devanagari_paired_test",
    "hinglish_paired_test",
    "devanagari_paired_dev",
    "hinglish_paired_dev",
    "devanagari_50",
    "hinglish_50",
]


def load_params():
    params = dict(DEFAULT_PARAMS)
    if BEST_PARAMS.exists():
        params.update(json.loads(BEST_PARAMS.read_text(encoding="utf-8")))
    return params


def count_changes(recipe, backbone, params, best):
    """How many settings differ from the main run (fixed recipe, hing-mbert, best params)."""
    changes = []
    if recipe != REFERENCE_RECIPE:
        changes.append(f"recipe={recipe}")
    if backbone != BACKBONE:
        changes.append(f"backbone={backbone}")
    for key, value in best.items():
        if params[key] != value:
            changes.append(f"{key}={params[key]}")
    return changes


class JointModel(nn.Module):
    def __init__(self, backbone, n_intents, n_tags, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
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


def slot_class_weights(examples, tag_names):
    """Rare tags get bigger weights: weight = 1 / sqrt(count), scaled so the average is 1."""
    counts = {}
    for name in tag_names:
        counts[name] = 1
    for example in examples:
        for tag in example["tags"]:
            counts[tag] += 1
    weights = []
    for name in tag_names:
        weights.append(1.0 / (counts[name] ** 0.5))
    mean = sum(weights) / len(weights)
    scaled = []
    for w in weights:
        scaled.append(w / mean)
    return scaled


def predict(model, rows, examples, tokenizer, intent_names, tag_names, device, batch_size):
    """Guess intent (with probabilities) and one tag per word for every example."""
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
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
                        "words": example["words"],
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


def tuning_score(dev, hinglish_dev):
    """What Optuna maximises: intent accuracy + slot F1, averaged over Devanagari and Hinglish dev."""
    total = dev["intent_accuracy"] + dev["slot_f1"]
    total += hinglish_dev["intent_accuracy"] + hinglish_dev["slot_f1"]
    return total / 4


def train(recipe, seed, backbone, params=None, out=None, trial=None, save_model=None):
    """Train one model. Returns the tuning score, or None if Optuna stopped the trial early."""
    if params is None:
        params = load_params()
        changes = count_changes(recipe, backbone, params, params)
        if len(changes) > 1:
            raise SystemExit(f"an ablation may change ONE setting from the main run, this changes {changes}")
    random.seed(seed)
    torch.manual_seed(seed)
    on_gpu = torch.cuda.is_available()
    device = "cuda" if on_gpu else "cpu"

    examples = training_examples(recipe)
    tests = test_sets()
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    if not on_gpu:
        # No GPU: tiny model, 20 examples, 2 steps - only proves the loop runs end to end.
        backbone = TINY_BACKBONE
        examples = examples[:20]
        small = {}
        for name, items in tests.items():
            small[name] = items[:20]
        tests = small
        epochs = 1
        batch_size = 10
    piece = max(1, batch_size // params["grad_accum"])  # rows per forward pass

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

    model = JointModel(backbone, len(intent_names), len(tag_names), params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    total_steps = epochs * ((len(train_rows) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(params["warmup"] * total_steps), total_steps)
    intent_loss_fn = nn.CrossEntropyLoss()  # skips every label that is -100
    if params["rare_slot_weighting"]:
        weights = torch.tensor(slot_class_weights(examples, tag_names), device=device)
        slot_loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        slot_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)  # mixed precision: faster on a T4

    if out is None:
        out = Path("artifacts") / "train" / f"{recipe}_{backbone.split('/')[-1]}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    curve_lines = ["epoch,train_loss,dev_intent_accuracy,dev_slot_f1,hinglish_dev_intent_accuracy,"
                   "hinglish_dev_slot_f1,seconds\n"]
    print(f"backbone {backbone} | recipe {recipe} | {len(train_rows):,} training sentences | device {device}")
    print(f"settings {params}")

    final = None
    for epoch in range(1, epochs + 1):
        started = time.time()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            optimizer.zero_grad()
            for part_start in range(0, len(batch_ids), piece):
                part_rows = []
                for i in batch_ids[part_start : part_start + piece]:
                    part_rows.append(train_rows[i])
                batch = make_batch(part_rows, tokenizer.pad_token_id, device)
                share = len(part_rows) / len(batch_ids)  # so the parts add up to one batch's loss
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=on_gpu):
                    intent_logits, slot_logits = model(batch["input_ids"], batch["attention_mask"])
                    intent_loss = intent_loss_fn(intent_logits, batch["intents"])
                    slot_loss = slot_loss_fn(slot_logits.reshape(-1, len(tag_names)), batch["tags"].reshape(-1))
                    loss = (intent_loss + params["slot_weight"] * slot_loss) * share
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            steps += 1
            if not on_gpu and steps == 2:
                break

        dev = score(predict(model, test_rows["dev"], tests["dev"], tokenizer, intent_names, tag_names,
                            device, batch_size))
        hinglish_dev = score(predict(model, test_rows["hinglish_paired_dev"], tests["hinglish_paired_dev"],
                                     tokenizer, intent_names, tag_names, device, batch_size))
        seconds = round(time.time() - started)
        print(
            f"epoch {epoch}: train loss {loss_sum / steps:.3f} | dev intent {dev['intent_accuracy']:.1%}"
            f" slot F1 {dev['slot_f1']:.1%} | hinglish dev intent {hinglish_dev['intent_accuracy']:.1%}"
            f" slot F1 {hinglish_dev['slot_f1']:.1%} | {seconds}s"
        )
        curve_lines.append(
            f"{epoch},{loss_sum / steps:.4f},{dev['intent_accuracy']:.4f},{dev['slot_f1']:.4f},"
            f"{hinglish_dev['intent_accuracy']:.4f},{hinglish_dev['slot_f1']:.4f},{seconds}\n"
        )
        final = tuning_score(dev, hinglish_dev)
        if trial is not None:
            trial.report(final, epoch)
            if trial.should_prune():
                return None
    (out / "curves.csv").write_text("".join(curve_lines), encoding="utf-8")
    if trial is not None:
        return final  # tuning trials keep only the score

    metrics = {"recipe": recipe, "seed": seed, "backbone": backbone, "on_gpu": on_gpu, "params": params}
    for name in SCORED_SETS:
        results = predict(model, test_rows[name], tests[name], tokenizer, intent_names, tag_names,
                          device, batch_size)
        write_jsonl(out / f"predictions_{name}.jsonl", results)
        metrics[name] = score(results)
        print(
            f"{name:24} intent {metrics[name]['intent_accuracy']:.1%}"
            f" | slot F1 {metrics[name]['slot_f1']:.1%} | n={metrics[name]['n']}"
        )
    metrics["intent_names"] = intent_names
    metrics["tag_names"] = tag_names
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if save_model is None:
        save_model = recipe == REFERENCE_RECIPE and backbone == BACKBONE
    if on_gpu and save_model:
        torch.save(model.state_dict(), out / "model.pt")
        tokenizer.save_pretrained(out / "tokenizer")
    print(f"saved everything in {out}/")
    return final
