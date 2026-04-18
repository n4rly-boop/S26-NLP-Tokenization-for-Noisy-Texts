"""
Train a single NER model on mixed-noise-augmented CoNLL-2003, then evaluate
on clean + OCR + ASR + social test sets. Designed as a subprocess per model
so MPS memory / process state is fully fresh per run.

Augmentation recipe:
    For each training sample, draw u ~ Uniform(0, 1).
      - if u < 0.5  -> keep clean
      - else        -> apply one of {OCR, ASR, social} noise (uniform).
    Per-sample RNG is seeded as (GLOBAL_SEED + sample_idx) for reproducibility.
    Validation and test sets stay clean (matches "train noisy, eval both
    clean and noisy" Scenario B).

Usage:
    python scripts/train_noisy_model.py <model_name>
        bert-base-uncased | bert-base-cased | gpt2 | google/byt5-small
"""
from __future__ import annotations

import gc
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    GPT2ForTokenClassification,
    T5Config,
    Trainer,
    TrainingArguments,
)

from noise import apply_asr_noise, apply_ocr_noise, apply_social_noise
from train import (
    ByT5ForTokenClassification,
    make_compute_metrics,
    tokenize_and_align_labels,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LABEL_NAMES = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_NAMES)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINTS_DIR = os.path.join(ROOT, "results", "checkpoints")
NOISY_DIR = os.path.join(ROOT, "data", "noisy")
PARTIALS_DIR = os.path.join(ROOT, "results", "tables", "noisy_training_partials")

GLOBAL_SEED = 1337
NOISE_P = 0.3  # same per-token noise rate used to generate data/noisy/* (see 02)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Mixed-noise augmentation
# ---------------------------------------------------------------------------


NOISE_FNS = {
    "ocr": apply_ocr_noise,
    "asr": apply_asr_noise,
    "social": apply_social_noise,
}


def augment_mixed_noise(train_ds: Dataset) -> Dataset:
    """
    Return a new Dataset of the same size where each sample is either
    clean (p=0.5) or noised with one of {OCR, ASR, social} (uniform, p=0.5).
    Per-sample seed = GLOBAL_SEED + idx so re-runs are reproducible.
    """
    noise_types = list(NOISE_FNS.keys())

    new_tokens, new_tags = [], []
    for idx, row in enumerate(train_ds):
        rng = random.Random(GLOBAL_SEED + idx)
        tokens = list(row["tokens"])
        tags = list(row["ner_tags"])

        if rng.random() < 0.5:
            # Clean
            new_tokens.append(tokens)
            new_tags.append(tags)
            continue

        noise = rng.choice(noise_types)
        fn = NOISE_FNS[noise]
        out_tokens, out_tags = fn(
            tokens, tags, ID2LABEL, LABEL2ID,
            p=NOISE_P, seed=GLOBAL_SEED + idx,
        )
        new_tokens.append(out_tokens)
        new_tags.append(out_tags)

    return Dataset.from_dict({"tokens": new_tokens, "ner_tags": new_tags})


# ---------------------------------------------------------------------------
# ByT5 tokenization
# ---------------------------------------------------------------------------


def tokenize_and_align_labels_byt5(examples, tokenizer, max_length=256):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    for tokens, ner_tags in zip(examples["tokens"], examples["ner_tags"]):
        input_ids = [0]
        label_ids = [-100]
        for word, label in zip(tokens, ner_tags):
            word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            input_ids += word_ids
            label_ids += [label] + [-100] * (len(word_ids) - 1)
        input_ids = input_ids[:max_length]
        label_ids = label_ids[:max_length]
        pad_len = max_length - len(input_ids)
        all_input_ids.append(input_ids + [0] * pad_len)
        all_attention_masks.append([1] * len(input_ids) + [0] * pad_len)
        all_labels.append(label_ids + [-100] * pad_len)
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Model loader (fresh pretrained weights)
# ---------------------------------------------------------------------------


def load_fresh_model_and_tokenizer(name: str):
    if name == "bert-base-uncased":
        model = BertForTokenClassification.from_pretrained(
            name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        )
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        return model, tok, tokenize_and_align_labels, {
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
        }
    if name == "bert-base-cased":
        model = BertForTokenClassification.from_pretrained(
            name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        )
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        return model, tok, tokenize_and_align_labels, {
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
        }
    if name == "gpt2":
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        tok.pad_token = tok.eos_token
        model = GPT2ForTokenClassification.from_pretrained(
            name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        )
        model.config.pad_token_id = tok.eos_token_id
        return model, tok, tokenize_and_align_labels, {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
        }
    if name == "google/byt5-small":
        model = ByT5ForTokenClassification.from_pretrained_encoder(name, NUM_LABELS)
        model.config.id2label = ID2LABEL
        model.config.label2id = LABEL2ID
        tok = AutoTokenizer.from_pretrained(name)
        return model, tok, tokenize_and_align_labels_byt5, {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "learning_rate": 3e-4,
            "gradient_accumulation_steps": 4,
            "tokenize_kwargs": {"max_length": 256},
        }
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on(trainer, tokenizer, tokenize_fn, ds_split, tokenize_kwargs=None):
    """Evaluate `trainer.model` on a raw word-level dataset split."""
    kwargs = tokenize_kwargs or {}
    tokenized = ds_split.map(
        lambda ex: tokenize_fn(ex, tokenizer, **kwargs),
        batched=True,
        remove_columns=ds_split.column_names,
    )
    trainer.eval_dataset = tokenized
    trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
    m = trainer.evaluate()
    return {
        "f1": round(float(m["eval_f1"]), 4),
        "precision": round(float(m["eval_precision"]), 4),
        "recall": round(float(m["eval_recall"]), 4),
    }


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_name>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    safe = model_name.replace("/", "__")
    print(f"[train_noisy_model] device={DEVICE} model={model_name}", flush=True)

    set_global_seed(GLOBAL_SEED)

    os.makedirs(PARTIALS_DIR, exist_ok=True)
    out_ckpt = os.path.join(CHECKPOINTS_DIR, f"{safe}-noisy")
    os.makedirs(out_ckpt, exist_ok=True)

    # Load clean CoNLL
    clean = load_dataset("conll2003", trust_remote_code=True)
    print(f"  CoNLL clean loaded: {len(clean['train'])} train / "
          f"{len(clean['validation'])} val / {len(clean['test'])} test",
          flush=True)

    # Mixed-noise augmentation on train split
    print("  augmenting train set (p=0.5 mixed OCR/ASR/social)...", flush=True)
    train_aug = augment_mixed_noise(clean["train"])
    print(f"  augmented train size: {len(train_aug)}", flush=True)

    # Load model + tokenizer
    model, tokenizer, tokenize_fn, cfg = load_fresh_model_and_tokenizer(model_name)
    tokenize_kwargs = cfg.pop("tokenize_kwargs", {}) if "tokenize_kwargs" in cfg else {}

    # Tokenize augmented train + clean validation
    print("  tokenizing train/validation...", flush=True)
    tokenized_train = train_aug.map(
        lambda ex: tokenize_fn(ex, tokenizer, **tokenize_kwargs),
        batched=True,
        remove_columns=train_aug.column_names,
    )
    tokenized_val = clean["validation"].map(
        lambda ex: tokenize_fn(ex, tokenizer, **tokenize_kwargs),
        batched=True,
        remove_columns=clean["validation"].column_names,
    )

    # Trainer
    bs_train = cfg["per_device_train_batch_size"]
    bs_eval = cfg["per_device_eval_batch_size"]
    lr = cfg["learning_rate"]
    grad_accum = cfg["gradient_accumulation_steps"]

    training_args = TrainingArguments(
        output_dir=out_ckpt,
        num_train_epochs=1,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="no",        # manual test-time evals
        save_strategy="epoch",
        logging_steps=200,
        fp16=(DEVICE == "cuda"),
        report_to="none",
        seed=GLOBAL_SEED,
        data_seed=GLOBAL_SEED,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if model_name.startswith("google/byt5") else None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(LABEL_NAMES),
    )

    print("  training on mixed-noise CoNLL...", flush=True)
    trainer.train()
    print("  training done.", flush=True)

    # Evaluate on clean + 3 noisy test sets
    results = []

    print("  evaluating on clean test...", flush=True)
    r = evaluate_on(trainer, tokenizer, tokenize_fn, clean["test"], tokenize_kwargs)
    r.update({"model": model_name, "noise": "clean"})
    results.append(r)
    print(f"    -> f1={r['f1']}", flush=True)

    for noise in ("ocr", "asr", "social"):
        print(f"  evaluating on noisy/{noise}/test...", flush=True)
        ds = load_from_disk(os.path.join(NOISY_DIR, noise))
        r = evaluate_on(trainer, tokenizer, tokenize_fn, ds["test"], tokenize_kwargs)
        r.update({"model": model_name, "noise": noise})
        results.append(r)
        print(f"    -> f1={r['f1']}", flush=True)

    out_path = os.path.join(PARTIALS_DIR, f"{safe}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)

    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
