"""
Tokenizer analysis metrics.

Functions:
    compute_tokenizer_stats  — fertility, OOV rate, UNK rate for a tokenizer
    print_stats_table        — pretty-print comparison across tokenizers
"""

from __future__ import annotations

from typing import Any


def compute_tokenizer_stats(
    tokenizer: Any,
    sentences: list[list[str]],
    sample_size: int = 2000,
) -> dict[str, float]:
    """
    Compute tokenization statistics over a list of pre-tokenized sentences.

    Parameters
    ----------
    tokenizer   : HuggingFace tokenizer (any type)
    sentences   : list of word lists, e.g. [["EU", "rejects", "German"], ...]
    sample_size : max number of sentences to use (for speed)

    Returns
    -------
    dict with keys:
        fertility   — avg number of tokens produced per input word
        oov_rate    — fraction of words that contain at least one UNK token
        unk_rate    — fraction of all tokens that are UNK
        vocab_size  — tokenizer vocabulary size (or None for byte-level)
    """
    sentences = sentences[:sample_size]

    total_words = 0
    total_tokens = 0
    unk_words = 0
    unk_tokens = 0

    unk_id = getattr(tokenizer, "unk_token_id", None)
    # ByT5 has no UNK — every byte is representable
    has_unk = unk_id is not None

    for words in sentences:
        total_words += len(words)

        # Tokenize the full sentence at once (faster than word-by-word)
        # Some tokenizers (ByT5, GPT-2) don't support is_split_into_words,
        # fall back to joining words into a plain string.
        try:
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=True,
                max_length=512,
            )
        except (ValueError, TypeError):
            encoding = tokenizer(
                " ".join(words),
                add_special_tokens=False,
                truncation=True,
                max_length=512,
            )
        ids = encoding["input_ids"]
        total_tokens += len(ids)

        if has_unk:
            unk_tokens += ids.count(unk_id)

            # Count how many individual words produce an UNK
            for word in words:
                word_ids = tokenizer(
                    word,
                    add_special_tokens=False,
                )["input_ids"]
                if unk_id in word_ids:
                    unk_words += 1

    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else None

    return {
        "fertility": round(total_tokens / total_words, 3) if total_words else 0.0,
        "oov_rate": round(unk_words / total_words, 4) if (total_words and has_unk) else 0.0,
        "unk_rate": round(unk_tokens / total_tokens, 4) if (total_tokens and has_unk) else 0.0,
        "vocab_size": vocab_size,
    }


def print_stats_table(results: dict[str, dict]) -> None:
    """
    Print a comparison table.

    Parameters
    ----------
    results : dict mapping tokenizer_name → stats dict from compute_tokenizer_stats
    """
    header = f"{'Tokenizer':<20} {'Vocab Size':>12} {'Fertility':>10} {'OOV Rate':>10} {'UNK Rate':>10}"
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        vocab = str(stats["vocab_size"]) if stats["vocab_size"] else "N/A (byte)"
        print(
            f"{name:<20} {vocab:>12} "
            f"{stats['fertility']:>10.3f} "
            f"{stats['oov_rate']:>10.4f} "
            f"{stats['unk_rate']:>10.4f}"
        )
