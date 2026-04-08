# NLP Case Study: Tokenization for Noisy Texts

**Course:** Natural Language Processing, Innopolis University, Spring 2026  
**Topic:** 1.7 — Tokenization for Noisy Texts  
**Deliverable:** GitHub repo + poster (presented at Final Exam)

---

## Overview

Tokenization is the first step in any NLP pipeline, and its behavior under noisy input is often overlooked.
This case study benchmarks three tokenization strategies on Named Entity Recognition (NER),
evaluating how each degrades under three types of real-world noise — then explores methods to mitigate the impact.

---

## Research Questions

1. How do different tokenizers behave on clean text? (fertility, OOV rate)
2. How does tokenizer quality degrade under OCR, ASR, and social media noise?
3. What pre-processing strategies recover the most performance?

---

## Noise Types

| Type | Examples |
|------|---------|
| **OCR** | `hel1o w0rld`, `thecat sat`, `ﬁle` (ligature errors) |
| **ASR** | `i wanna go to new york tommorow`, missing punctuation, homophones |
| **Social media** | `omg wtf lol`, `#NLP`, `@user`, emojis, ALL CAPS, elongation (`sooooo`) |

---

## Tokenizers Compared

| Tokenizer | Strategy | Model |
|-----------|----------|-------|
| WordPiece | Subword (frequency-based) | `bert-base-uncased` |
| BPE | Byte Pair Encoding | `gpt2` |
| Byte-level | Raw bytes, no OOV possible | `google/byt5-small` |

---

## Downstream Task

**NER on CoNLL-2003** — standard sequence labeling benchmark.  
Labels: `PER`, `ORG`, `LOC`, `MISC`, `O`  
Metric: **seqeval F1** (entity-level)

---

## Experiment Plan

### Phase 1 — Baseline (clean data)
- Tokenizer analysis: fertility, OOV rate, vocab coverage
- Fine-tune NER on clean CoNLL-2003 train set
- Evaluate on clean test set → reference F1 per tokenizer

### Phase 2 — Noisy evaluation
- Synthesize OCR / ASR / social noise on CoNLL-2003
- Re-evaluate all three models (no retraining)
- Measure F1 degradation per noise type

### Phase 3 — Improvements
- Apply pre-processing methods (spell correction, text normalization)
- Re-evaluate → measure recovery
- Compare tokenizers after fix

---

## Repo Structure

```
├── data/
│   ├── raw/              # CoNLL-2003 (downloaded via script)
│   └── noisy/            # Synthesized noisy versions
│       ├── ocr/
│       ├── asr/
│       └── social/
├── notebooks/
│   ├── 01_baseline.ipynb         # Clean data: tokenizer stats + NER training
│   ├── 02_noise_synthesis.ipynb  # Generate and inspect noisy datasets
│   ├── 03_noisy_evaluation.ipynb # Evaluate baseline models on noisy data
│   └── 04_improvements.ipynb     # Apply fixes, final comparison
├── src/
│   ├── metrics.py   # Tokenizer stats (fertility, OOV) + seqeval wrapper
│   └── train.py     # NER training utilities, model wrappers
├── results/
│   └── tables/      # CSV results for each experiment phase
├── poster/
│   └── poster.pdf   # Final poster (motivation, methodology, results, conclusions)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/01_baseline.ipynb
```

---

## Results

*To be filled after experiments.*

| Tokenizer | Fertility | OOV% | F1 (clean) | F1 (OCR) | F1 (ASR) | F1 (social) |
|-----------|-----------|------|------------|----------|----------|-------------|
| BERT WordPiece | — | — | — | — | — | — |
| GPT-2 BPE | — | — | — | — | — | — |
| ByT5 Byte-level | — | — | — | — | — | — |
