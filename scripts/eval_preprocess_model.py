"""
Evaluate a single model on preprocessed noisy test sets (OCR, ASR, social).

Designed to be invoked as a subprocess so MPS / process memory is fully
released between models. Writes one JSON record per (model, noise) to
results/tables/preprocess_partials/<model>.json.

Usage:
    python scripts/eval_preprocess_model.py <model_name>

where <model_name> is one of:
    bert-base-uncased, bert-base-cased, gpt2, google/byt5-small
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    GPT2ForTokenClassification,
    T5Config,
    Trainer,
    TrainingArguments,
)

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
NUM_LABELS = len(LABEL_NAMES)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINTS_DIR = os.path.join(ROOT, "results", "checkpoints")
PREPROC_DIR = os.path.join(ROOT, "data", "preprocessed")
PARTIALS_DIR = os.path.join(ROOT, "results", "tables", "preprocess_partials")

NOISE_TYPES = ["ocr", "asr", "social"]

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# ByT5 tokenization (custom, no word_ids)
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
# Loader
# ---------------------------------------------------------------------------


def find_best_checkpoint(model_key: str) -> str:
    model_dir = os.path.join(CHECKPOINTS_DIR, model_key)
    cps = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*")))
    if not cps:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    return cps[-1]


def load_model_and_tokenizer(name: str):
    if name == "bert-base-uncased":
        ckpt = find_best_checkpoint("bert-base-uncased")
        model = BertForTokenClassification.from_pretrained(ckpt)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
        return model, tok, tokenize_and_align_labels, 32, None
    if name == "bert-base-cased":
        ckpt = os.path.join(CHECKPOINTS_DIR, "bert-base-cased", "checkpoint-878")
        model = BertForTokenClassification.from_pretrained(ckpt)
        tok = AutoTokenizer.from_pretrained("bert-base-cased", add_prefix_space=True)
        return model, tok, tokenize_and_align_labels, 32, None
    if name == "gpt2":
        ckpt = find_best_checkpoint("gpt2")
        model = GPT2ForTokenClassification.from_pretrained(ckpt)
        tok = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
        tok.pad_token = tok.eos_token
        return model, tok, tokenize_and_align_labels, 16, None
    if name == "google/byt5-small":
        ckpt = find_best_checkpoint("byt5")
        config = T5Config.from_pretrained("google/byt5-small")
        model = ByT5ForTokenClassification(config, num_labels=NUM_LABELS)
        _bin = os.path.join(ckpt, "pytorch_model.bin")
        _sft = os.path.join(ckpt, "model.safetensors")
        if os.path.exists(_bin):
            state = torch.load(_bin, map_location="cpu")
            model.load_state_dict(state, strict=False)
        elif os.path.exists(_sft):
            from safetensors.torch import load_file as _load_sf
            state = _load_sf(_sft)
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"No weights in {ckpt}")
        tok = AutoTokenizer.from_pretrained("google/byt5-small")
        return model, tok, tokenize_and_align_labels_byt5, 2, {"max_length": 256}
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def eval_one(model, tokenizer, tokenize_fn, noise_type, model_name,
             per_device_eval_batch_size=32, tokenize_kwargs=None):
    ds = load_from_disk(os.path.join(PREPROC_DIR, noise_type))
    kwargs = tokenize_kwargs or {}
    tokenized = ds["test"].map(
        lambda ex: tokenize_fn(ex, tokenizer, **kwargs),
        batched=True,
        remove_columns=ds["test"].column_names,
    )
    eval_args = TrainingArguments(
        output_dir="/tmp/eval_tmp",
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=(DEVICE == "cuda"),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(LABEL_NAMES),
    )
    trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
    m = trainer.evaluate()
    out = {
        "model": model_name,
        "noise": noise_type,
        "f1": round(float(m["eval_f1"]), 4),
        "precision": round(float(m["eval_precision"]), 4),
        "recall": round(float(m["eval_recall"]), 4),
    }
    del trainer, tokenized
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_name>", file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    os.makedirs(PARTIALS_DIR, exist_ok=True)

    print(f"[eval_preprocess_model] device={DEVICE} model={model_name}", flush=True)

    model, tok, tfn, bs, tkw = load_model_and_tokenizer(model_name)
    print("  loaded.", flush=True)

    results = []
    for noise in NOISE_TYPES:
        print(f"  eval on preprocessed/{noise}...", flush=True)
        r = eval_one(model, tok, tfn, noise, model_name,
                     per_device_eval_batch_size=bs, tokenize_kwargs=tkw)
        print(f"    -> f1={r['f1']}", flush=True)
        results.append(r)

    # Safe filename
    safe = model_name.replace("/", "__")
    out_path = os.path.join(PARTIALS_DIR, f"{safe}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
