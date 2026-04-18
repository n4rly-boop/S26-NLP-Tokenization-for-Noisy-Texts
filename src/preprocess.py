"""
Inference-time preprocessing for noisy CoNLL-2003 NER.

Word-level preprocessors per noise channel. All functions have signature
    (tokens: List[str]) -> List[str]
and MUST preserve list length so BIO label alignment is unaffected.

Components
----------
OCR     : char de-confusion (reverse OCR_CONFUSIONS), spell correction (pyspellchecker)
ASR     : truecasing via CoNLL-train-derived dict, unambiguous homophone fix
Social  : collapse repeated chars, reverse SOCIAL_ABBREV, spell correction

References
----------
- OCR char map is the reverse of the forward map in src/noise.py
  (Gui et al. / TextFlint, 2021).
- Homophone reversal is kept intentionally *narrow* to unambiguous slang
  short-forms; full PHONETIC_MAP is ambiguous (e.g. their <-> there).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

from noise import OCR_CONFUSIONS, SOCIAL_ABBREV

# ---------------------------------------------------------------------------
# Lazy-initialized spellchecker (pyspellchecker is slow to import)
# ---------------------------------------------------------------------------

_SPELL = None


def _get_spell():
    """Lazy init of pyspellchecker.SpellChecker."""
    global _SPELL
    if _SPELL is None:
        from spellchecker import SpellChecker
        # distance=1 is ~10-100x faster than default distance=2 and still catches
        # the vast majority of single-typo errors produced by our noise channels.
        _SPELL = SpellChecker(distance=1)
    return _SPELL


@lru_cache(maxsize=200_000)
def _spell_correct_lower(low: str) -> Optional[str]:
    """
    Return spell-corrected form of a lowercase token, or None to keep original.

    Cached across the process — most tokens repeat, so this is the hot path.
    """
    if not low or not any(c.isalpha() for c in low):
        return None
    spell = _get_spell()
    if low in spell:
        return None
    try:
        corrected = spell.correction(low)
    except Exception:
        return None
    if corrected is None or " " in corrected or corrected == low:
        return None
    return corrected


def _spell_correct_token(tok: str) -> str:
    """
    Correct a single token via cached pyspellchecker lookup.

    Preserves original casing pattern (ALL-UPPER / Title-case / lower).
    """
    if not tok:
        return tok
    low = tok.lower()
    corrected = _spell_correct_lower(low)
    if corrected is None:
        return tok
    if tok.isupper():
        return corrected.upper()
    if tok[:1].isupper() and tok[1:].islower():
        return corrected.capitalize()
    return corrected


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

# Reverse OCR map: digit/symbol → plausible original letter.
# noise.py forward map is many-to-one (e.g. both 'o' and 'O' → '0'); for
# reversal we pick the *lowercase letter* variant as the canonical restore
# target. Spell-correction downstream repairs residual casing.
_OCR_REVERSE: Dict[str, str] = {
    '0': 'o', '1': 'l', '3': 'e', '5': 's', '@': 'a',
    '4': 'a', '9': 'g', '6': 'b', '8': 'b', '+': 't', '2': 'z',
}


def _token_has_digit_or_symbol(tok: str) -> bool:
    return any(ch in _OCR_REVERSE for ch in tok)


def _token_is_pure_number(tok: str) -> bool:
    """True if token is digits (optionally with , . -) — leave numbers alone."""
    if not tok:
        return False
    return all(ch.isdigit() or ch in ",.-" for ch in tok)


def ocr_char_fix(tokens: List[str]) -> List[str]:
    """
    Reverse OCR character confusions.

    Conservative: only apply if token mixes letters and digits/symbols.
    Pure-number tokens (scores, years, counts) are preserved.
    """
    out: List[str] = []
    for tok in tokens:
        if not tok or _token_is_pure_number(tok) or not _token_has_digit_or_symbol(tok):
            out.append(tok)
            continue
        # Mixed token: reverse known confusions char-by-char
        fixed = "".join(_OCR_REVERSE.get(ch, ch) for ch in tok)
        out.append(fixed)
    assert len(out) == len(tokens)
    return out


def ocr_spell_correct(tokens: List[str]) -> List[str]:
    """Per-token spellchecker pass (skip in-dict tokens)."""
    out = [_spell_correct_token(t) for t in tokens]
    assert len(out) == len(tokens)
    return out


def ocr_pipeline(tokens: List[str]) -> List[str]:
    out = ocr_spell_correct(ocr_char_fix(tokens))
    assert len(out) == len(tokens)
    return out


# ---------------------------------------------------------------------------
# ASR — truecasing + homophone fix
# ---------------------------------------------------------------------------

# Unambiguous homophone / slang short-form reverse map.
# Keys are lowercase noisy forms; values are their restored canonical form.
# NOTE: ambiguous pairs (their/there, its/it's, than/then, by/bye, etc.) are
# deliberately EXCLUDED — reversing them would risk introducing errors.
_HOMOPHONE_REVERSE: Dict[str, str] = {
    "ur": "your",
    "u": "you",
    "4": "for",
    "r": "are",
    "bee": "be",
    "c": "see",
    "y": "why",
    "2": "to",
}

# Module-level truecase dict — populated by build_truecase_dict() or
# load_truecase_dict(). Keyed by lowercase word.
_TRUECASE_DICT: Dict[str, str] = {}


def build_truecase_dict(train_dataset) -> Dict[str, str]:
    """
    Build a lowercase-word -> most-common-cased-form dict from CoNLL train.

    `train_dataset` is a HuggingFace Dataset with a `tokens` column (List[str]).

    Side-effect: updates the module-level _TRUECASE_DICT used by asr_truecase().
    """
    counter: Dict[str, Counter] = defaultdict(Counter)
    for row in train_dataset:
        for tok in row["tokens"]:
            if not tok:
                continue
            counter[tok.lower()][tok] += 1

    mapping: Dict[str, str] = {
        low: cnt.most_common(1)[0][0] for low, cnt in counter.items()
    }

    global _TRUECASE_DICT
    _TRUECASE_DICT = mapping
    return mapping


def load_truecase_dict(path: str) -> Dict[str, str]:
    """Load truecase dict from JSON and set it as the module default."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    global _TRUECASE_DICT
    _TRUECASE_DICT = mapping
    return mapping


def save_truecase_dict(mapping: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)


def asr_truecase(tokens: List[str], mapping: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Restore casing from CoNLL-train dict.

    - Lookup lower(tok); if hit, use stored cased form.
    - Else keep token as-is (already lowercase after ASR noise pass).
    - Sentence-initial heuristic: if first alpha token has no lookup hit, Title-case it.
    """
    m = mapping if mapping is not None else _TRUECASE_DICT

    out: List[str] = []
    first_alpha_done = False
    for tok in tokens:
        if not tok:
            out.append(tok)
            continue

        low = tok.lower()
        if low in m:
            out.append(m[low])
        else:
            if not first_alpha_done and any(c.isalpha() for c in tok):
                out.append(tok[:1].upper() + tok[1:])
            else:
                out.append(tok)

        if not first_alpha_done and any(c.isalpha() for c in tok):
            first_alpha_done = True

    assert len(out) == len(tokens)
    return out


def asr_homophone_fix(tokens: List[str]) -> List[str]:
    """Reverse unambiguous slang/phonetic shorthand."""
    out: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _HOMOPHONE_REVERSE:
            rep = _HOMOPHONE_REVERSE[low]
            # Preserve capitalization where reasonable
            if tok[:1].isupper():
                rep = rep[:1].upper() + rep[1:]
            out.append(rep)
        else:
            out.append(tok)
    assert len(out) == len(tokens)
    return out


def asr_pipeline(tokens: List[str]) -> List[str]:
    out = asr_homophone_fix(asr_truecase(tokens))
    assert len(out) == len(tokens)
    return out


# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------

_REPEAT_RE = re.compile(r"(.)\1{2,}")

# Reverse of SOCIAL_ABBREV: noisy form → original (e.g. 'gr8' → 'great').
_SOCIAL_ABBREV_REVERSE: Dict[str, str] = {v: k for k, v in SOCIAL_ABBREV.items()}


def social_unrepeat(tokens: List[str]) -> List[str]:
    """Collapse runs of 3+ identical chars down to 2 (e.g. 'cooool' -> 'cool')."""
    out = [_REPEAT_RE.sub(r"\1\1", t) for t in tokens]
    assert len(out) == len(tokens)
    return out


def social_unabbrev(tokens: List[str]) -> List[str]:
    """Reverse SOCIAL_ABBREV shortforms (case-insensitive match, preserve first-char case)."""
    out: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _SOCIAL_ABBREV_REVERSE:
            rep = _SOCIAL_ABBREV_REVERSE[low]
            if tok[:1].isupper():
                rep = rep[:1].upper() + rep[1:]
            out.append(rep)
        else:
            out.append(tok)
    assert len(out) == len(tokens)
    return out


def social_spell_correct(tokens: List[str]) -> List[str]:
    """Same pyspellchecker pass as OCR."""
    out = [_spell_correct_token(t) for t in tokens]
    assert len(out) == len(tokens)
    return out


def social_pipeline(tokens: List[str]) -> List[str]:
    out = social_spell_correct(social_unabbrev(social_unrepeat(tokens)))
    assert len(out) == len(tokens)
    return out
