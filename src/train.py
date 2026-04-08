"""
NER training utilities.

Supports:
    - BERT-style models (BertForTokenClassification)
    - GPT-2 (GPT2ForTokenClassification)
    - ByT5 encoder (custom ByT5ForTokenClassification)

Public API:
    get_model_and_tokenizer  — factory: returns (model, tokenizer) for a given hub name
    tokenize_and_align_labels — tokenize CoNLL examples, align NER labels to subword tokens
    make_compute_metrics      — returns a seqeval compute_metrics closure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    GPT2ForTokenClassification,
    T5EncoderModel,
    PreTrainedModel,
    modeling_outputs,
)

# ──────────────────────────────────────────────────────────────────────────────
# ByT5 custom model
# ──────────────────────────────────────────────────────────────────────────────


class ByT5ForTokenClassification(PreTrainedModel):
    """
    T5 encoder + linear classification head for token-level tasks.

    Works with google/byt5-small (and any T5-encoder checkpoint).
    """

    def __init__(self, config, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels
        self.encoder = T5EncoderModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.d_model, num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> modeling_outputs.TokenClassifierOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained_encoder(
        cls, model_name: str, num_labels: int
    ) -> "ByT5ForTokenClassification":
        from transformers import T5Config
        config = T5Config.from_pretrained(model_name)
        model = cls(config, num_labels=num_labels)
        # Load pretrained encoder weights
        pretrained = T5EncoderModel.from_pretrained(model_name)
        model.encoder.load_state_dict(pretrained.state_dict())
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "bert-base-uncased": "bert",
    "gpt2": "gpt2",
    "google/byt5-small": "byt5",
}


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
) -> tuple[PreTrainedModel, Any]:
    """
    Load model + tokenizer for NER fine-tuning.

    Parameters
    ----------
    model_name  : HuggingFace hub name (must be in SUPPORTED_MODELS)
    num_labels  : number of NER label classes
    id2label    : optional label id → name mapping
    label2id    : optional label name → id mapping

    Returns
    -------
    (model, tokenizer)
    """
    model_type = SUPPORTED_MODELS.get(model_name)
    if model_type is None:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(SUPPORTED_MODELS)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    if model_type == "bert":
        model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

    elif model_type == "gpt2":
        # GPT-2 has no pad token — use eos
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model.config.pad_token_id = tokenizer.eos_token_id

    elif model_type == "byt5":
        model = ByT5ForTokenClassification.from_pretrained_encoder(
            model_name, num_labels=num_labels
        )
        if id2label:
            model.config.id2label = id2label
        if label2id:
            model.config.label2id = label2id

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset tokenization
# ──────────────────────────────────────────────────────────────────────────────


def tokenize_and_align_labels(
    examples: dict,
    tokenizer: Any,
    label_column: str = "ner_tags",
    max_length: int = 512,
) -> dict:
    """
    Tokenize a batch of CoNLL examples and align NER labels to subword tokens.

    For multi-token words (e.g. "Washington" → ["Wash", "##ington"]):
      - First subtoken gets the original label
      - Remaining subtokens get -100 (ignored in CrossEntropyLoss)

    Works uniformly for WordPiece, BPE, and byte-level tokenizers.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    all_labels = []
    for i, labels in enumerate(examples[label_column]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], <pad>)
                aligned.append(-100)
            elif word_id != prev_word_id:
                # First subtoken of a word → use the real label
                aligned.append(labels[word_id])
            else:
                # Continuation subtoken → ignore
                aligned.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def make_compute_metrics(label_names: list[str]):
    """
    Returns a compute_metrics function compatible with HuggingFace Trainer.

    Uses seqeval for entity-level F1 (standard for CoNLL NER).
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            true_label_row, true_pred_row = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label == -100:
                    continue
                true_label_row.append(label_names[label])
                true_pred_row.append(label_names[pred])
            true_labels.append(true_label_row)
            true_preds.append(true_pred_row)

        return {
            "f1": f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
        }

    return compute_metrics
