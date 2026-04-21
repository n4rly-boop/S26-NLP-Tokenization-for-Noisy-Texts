# NLP Case Study: Tokenization for Noisy Texts

**Course:** Natural Language Processing, Innopolis University, Spring 2026
**Topic:** 1.7 — Tokenization for Noisy Texts
**Deliverable:** GitHub repo + poster (presented at Final Exam)

### 📄 [**View the poster (PDF)**](poster/poster.pdf)

> *Tokenization for Noisy Texts: An Empirical Study on Named Entity Recognition* —
> 4 tokenizers × 3 noise channels × 2 mitigations, with mechanism analysis (Pearson r = 0.88).

---

## Overview

Tokenization is the first step in any NLP pipeline, and its behavior under noisy input is often overlooked.
Modern subword/byte-level tokenizers avoid OOV by construction — but do they stay **stable** under noise?
This case study benchmarks four tokenization strategies on Named Entity Recognition (NER),
measures degradation under three real-world noise channels, and compares two mitigation strategies
(inference-time preprocessing vs noisy fine-tuning).

---

## Research Questions

1. How does noise perturb tokenization? (fertility inflation, token drift)
2. Which tokenizer family (WordPiece / BPE / byte-level) is most noise-robust?
3. Can inference-time preprocessing or noisy fine-tuning recover NER F1, and which works when?

---

## Noise Channels

| Channel | Examples |
|---------|----------|
| **OCR** | `hel1o w0rld`, visual-confusion substitutions, ligature errors |
| **ASR** | lowercase + no punctuation, homophone swaps (`to` → `too`), missing casing |
| **Social** | `omg wtf lol`, elongation (`sooooo`), abbreviations (`gr8`, `u`), typos |

All three applied at per-token probability `p = 0.3` (Belinkov & Bisk / TextFlint style).

---

## Models & Tokenizers

| Tokenizer | Strategy | Model | Vocab | Clean F1 |
|-----------|----------|-------|-------|----------|
| WordPiece | Subword | `bert-base-uncased` | 30.5K | 0.89 |
| WordPiece | Subword (cased) | `bert-base-cased` | 28.9K | 0.90 |
| BPE | Byte Pair Encoding | `gpt2` | 50.2K | 0.69 |
| Byte-level | Raw bytes | `google/byt5-small` | 256 | 0.86 |

---

## Downstream Task

**NER on CoNLL-2003 English** — 3,453 test sentences.
Labels: `PER`, `ORG`, `LOC`, `MISC`, `O`.
Metric: **seqeval F1** (entity-level).

---

## Mitigations

- **Inference preprocess** — per-channel rule-based pipelines (word-count preserving, so BIO labels stay aligned):
  - OCR: char-fix → spellcheck
  - ASR: truecase → homophone-fix
  - Social: un-repeat → un-abbrev → spellcheck
- **Noisy fine-tune** — 1 epoch, 50% clean / 50% mixed-noise training data.

---

## Results

**Mean F1 across 12 (model × noise) cells:**

| Stage | Mean F1 | % of noise gap recovered |
|-------|---------|--------------------------|
| Clean baseline | **0.83** | — |
| Noisy (no mitigation) | 0.55 (−0.28) | 0% |
| + inference preprocess | 0.69 | 49% |
| + noisy fine-tune | 0.69 | 47% |
| Best per cell | **0.71** | **55%** |

**Per-model F1 (noisy → best mitigation):**

| Model | OCR | ASR | Social |
|-------|-----|-----|--------|
| bert-base-uncased | 0.66 → 0.78 | 0.83 → 0.85 | 0.62 → 0.76 |
| bert-base-cased | 0.70 → 0.80 | 0.21 → 0.68 | 0.71 → 0.78 |
| gpt2 | 0.54 → 0.60 | 0.03 → 0.46 | 0.53 → 0.54 |
| byt5-small | 0.70 → 0.79 | 0.39 → 0.68 | 0.71 → 0.76 |

**Key findings:**

- "No UNK" ≠ noise-robust. Byte-level tokenization prevents fertility inflation but NER F1 still drops.
- Single component dominates per channel (truecase / ASR, charfix / OCR, spellcheck / social) — target, don't uniformly preprocess.
- Preprocess vs fine-tune is a per-cell tradeoff; no universal winner.
- Mechanism: both interventions act via **token-set stability** (Pearson r = 0.88 between Δtoken-overlap and ΔF1 across 28 LOO component cells).

Full per-cell tables live in `results/tables/` (see the poster for plots).

---

## Repo Structure

```
├── data/
│   ├── raw/              # CoNLL-2003 (downloaded via script, gitignored)
│   └── noisy/            # Synthesized noisy versions (gitignored)
│       ├── ocr/
│       ├── asr/
│       └── social/
├── notebooks/
│   ├── 01_baseline.ipynb              # Clean tokenizer stats + NER training (bert-uncased, gpt2, byt5)
│   ├── 02_noise_synthesis.ipynb       # Generate + inspect OCR/ASR/social noise
│   ├── 03_noisy_evaluation.ipynb      # Evaluate baseline models on noisy test sets
│   ├── 04_bert_cased.ipynb            # Cased-model ablation (casing as tokenization signal)
│   ├── 05_preprocess.ipynb            # Rule-based preprocessing per noise channel
│   ├── 06_noisy_training.ipynb        # Mixed-noise fine-tuning (train-time mitigation)
│   ├── 07_preprocess_ablation.ipynb   # Leave-one-out component attribution (F1)
│   └── 08_tokenizer_analysis.ipynb    # Tokenizer stats end-to-end: clean → noisy → preprocess → LOO
├── scripts/
│   ├── train_noisy_model.py           # Training entry point (subprocess per model, MPS-safe)
│   ├── eval_preprocess_model.py       # Preprocess-eval entry point
│   └── eval_ablation.py               # LOO ablation entry point
├── src/
│   ├── metrics.py                     # Tokenizer stats + seqeval wrapper
│   ├── train.py                       # NER training utilities, model wrappers
│   ├── preprocess.py                  # Word-count-preserving preprocessors (charfix, spellcheck, truecase, …)
│   └── noise.py                       # OCR / ASR / social perturbation functions
├── results/
│   └── tables/                        # CSV results (clean, noisy, preprocess, ablation, tokenizer)
├── poster/
│   ├── poster.tex                     # tikzposter source
│   ├── poster.pdf                     # Final poster (A1 landscape)
│   ├── make_poster_figures.py         # Regenerate all poster figures from results CSVs
│   └── figures/                       # Generated PDF figures
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/01_baseline.ipynb
```

Notebooks run top-to-bottom in order. Training steps use a subprocess-per-model pattern
(see `scripts/`) to avoid MPS memory fragmentation on Apple Silicon.

---

## References

- Belinkov & Bisk, *Synthetic and Natural Noise Both Break Neural Machine Translation*, ICLR 2018.
- Wang et al., *TextFlint: Unified Multilingual Robustness Evaluation Toolkit for NLP*, 2021.
- Tjong Kim Sang & De Meulder, *Introduction to the CoNLL-2003 Shared Task*, 2003.
- Nakayama, *seqeval: A Python framework for sequence labeling evaluation*.
- Xue et al., *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*, TACL 2022.
