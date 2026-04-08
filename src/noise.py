"""
Synthetic noise generation for CoNLL-2003 NER data.

Implements three noise channels simulating real-world text degradation:
  - OCR noise: visual character confusions (Gui et al. / TextFlint, 2021)
  - ASR noise: speech recognition errors and phonetic substitutions
  - Social media noise: keyboard typos, abbreviations, stylistic distortions

References:
  - Piktus et al. (2019). Misspelling Oblivious Word Embeddings.
    https://arxiv.org/abs/1905.09755
    (QWERTY keyboard adjacency for typo simulation)

  - Gui et al. / TextFlint (2021). TextFlint: Unified Multilingual Robustness
    Evaluation Toolkit for Natural Language Processing.
    https://arxiv.org/abs/2103.11441
    (OCR visual character confusion mappings)

  - Wei & Zou (2019). EDA: Easy Data Augmentation Techniques for Boosting
    Performance on Text Classification Tasks.
    https://arxiv.org/abs/1901.11196
    (General token-level augmentation methodology)
"""

import random
import string
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

OCR_CONFUSIONS: Dict[str, str] = {
    'o': '0', 'O': '0', 'l': '1', 'I': '1', 'i': '1',
    'e': '3', 'E': '3', 's': '5', 'S': '5', 'a': '@',
    'A': '4', 'g': '9', 'G': '6', 'b': '6', 'B': '8',
    't': '+', 'z': '2', 'Z': '2',
}

# Digraphs applied before single-character substitutions
OCR_DIGRAPHS: Dict[str, str] = {
    'rn': 'm',
    'vv': 'w',
}

QWERTY_NEIGHBORS: Dict[str, List[str]] = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'z', 'x'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
    'g': ['f', 't', 'y', 'h', 'v', 'b'], 'h': ['g', 'y', 'u', 'j', 'b', 'n'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k'],
}

PHONETIC_MAP: Dict[str, str] = {
    'their': 'there', 'there': 'their', "they're": 'there',
    'your': 'ur', "you're": 'ur', 'you': 'u',
    'are': 'r', 'for': '4', 'to': 'too', 'too': 'to',
    'two': '2', 'be': 'bee', 'see': 'c', 'why': 'y',
    'know': 'no', 'new': 'knew', 'one': 'won', 'won': 'one',
    'right': 'write', 'write': 'right', 'by': 'bye',
    'buy': 'by', 'hear': 'here', 'here': 'hear',
    'its': "it's", 'than': 'then', 'then': 'than',
}

SOCIAL_ABBREV: Dict[str, str] = {
    'great': 'gr8', 'before': 'b4', 'later': 'l8r',
    'because': 'bcuz', 'please': 'plz', 'thanks': 'thx',
    'people': 'ppl', 'something': 'smth', 'everyone': 'evry1',
    'tonight': '2nite', 'tomorrow': '2moro', 'today': '2day',
    'laughing': 'lol', 'love': 'luv', 'hate': 'h8',
}

FUNCTION_WORDS = {
    'the', 'a', 'an', 'of', 'in', 'at', 'on', 'is', 'was',
    'are', 'were', 'been', 'be', 'to', 'and', 'or', 'but',
    'it', 'its', 'this', 'that', 'with', 'as', 'by', 'from',
}

# ---------------------------------------------------------------------------
# BIO tag repair
# ---------------------------------------------------------------------------

def fix_bio_sequence(
    tags: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> List[int]:
    """
    Repair a BIO tag sequence after token deletions.

    Any I-X tag that follows O or a tag from a different entity type is
    converted to B-X so the sequence remains valid BIO.

    Parameters
    ----------
    tags     : list of integer tag ids
    id2label : mapping from tag id to label string (e.g. {2: 'I-PER'})
    label2id : mapping from label string to tag id

    Returns
    -------
    A new list of integer tag ids with invalid I- transitions corrected.
    """
    if not tags:
        return tags

    fixed: List[int] = []
    prev_label = 'O'

    for tag_id in tags:
        label = id2label[tag_id]
        parts = label.split('-', 1)

        if parts[0] == 'I':
            entity_type = parts[1]
            prev_parts = prev_label.split('-', 1)
            prev_prefix = prev_parts[0]
            prev_type = prev_parts[1] if len(prev_parts) > 1 else None

            # Invalid if previous was O or a different entity type
            if prev_prefix == 'O' or prev_type != entity_type:
                b_label = f'B-{entity_type}'
                tag_id = label2id[b_label]
                label = b_label

        fixed.append(tag_id)
        prev_label = label

    return fixed


# ---------------------------------------------------------------------------
# OCR noise
# ---------------------------------------------------------------------------

def _apply_ocr_digraphs(token: str) -> str:
    """Replace digraph patterns in token before per-character substitutions."""
    for digraph, replacement in OCR_DIGRAPHS.items():
        token = token.replace(digraph, replacement)
    return token


def _apply_ocr_confusions(token: str) -> str:
    """Replace individual characters with visually similar substitutes."""
    chars = []
    for ch in token:
        if ch in OCR_CONFUSIONS and random.random() < 0.6:
            chars.append(OCR_CONFUSIONS[ch])
        else:
            chars.append(ch)
    return ''.join(chars)


def apply_ocr_noise(
    tokens: List[str],
    tags: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    p: float = 0.3,
    seed: int = None,
) -> Tuple[List[str], List[int]]:
    """
    Simulate OCR-style visual character confusion noise.

    For each token, with probability *p*:
      1. Apply digraph substitutions (OCR_DIGRAPHS).
      2. Apply per-character substitutions (OCR_CONFUSIONS, prob 0.6 each).
      3. With prob 0.1: delete a random interior character.
      4. With prob 0.1: duplicate a random character.

    Additionally, with probability p*0.15, adjacent tokens are merged
    (concatenated strings, first token's tag kept, second deleted).
    fix_bio_sequence is called after any merges.

    Parameters
    ----------
    tokens    : word-level token list
    tags      : corresponding integer NER tag list
    id2label  : id -> label string mapping
    label2id  : label string -> id mapping
    p         : per-token noise probability (default 0.3)
    seed      : optional random seed for reproducibility

    Returns
    -------
    (noisy_tokens, noisy_tags)
    """
    if seed is not None:
        random.seed(seed)

    if not tokens:
        return tokens, tags

    out_tokens = list(tokens)
    out_tags = list(tags)

    # Per-token character-level noise
    for i in range(len(out_tokens)):
        if random.random() < p:
            tok = out_tokens[i]

            # Step 1: digraph substitutions
            tok = _apply_ocr_digraphs(tok)

            # Step 2: per-character confusion substitutions
            tok = _apply_ocr_confusions(tok)

            # Step 3: random interior character deletion
            if random.random() < 0.1 and len(tok) > 2:
                idx = random.randint(1, len(tok) - 2)
                tok = tok[:idx] + tok[idx + 1:]

            # Step 4: random character duplication
            if random.random() < 0.1 and len(tok) >= 1:
                idx = random.randint(0, len(tok) - 1)
                tok = tok[:idx] + tok[idx] + tok[idx:]

            out_tokens[i] = tok

    # Token-level merge pass (iterate backwards to preserve indices)
    merge_p = p * 0.15
    merged_tokens: List[str] = []
    merged_tags: List[int] = []
    skip_next = False

    for i in range(len(out_tokens)):
        if skip_next:
            skip_next = False
            continue

        if (
            i < len(out_tokens) - 1
            and random.random() < merge_p
        ):
            # Merge token[i] and token[i+1]
            merged_tokens.append(out_tokens[i] + out_tokens[i + 1])
            merged_tags.append(out_tags[i])
            skip_next = True
        else:
            merged_tokens.append(out_tokens[i])
            merged_tags.append(out_tags[i])

    if len(merged_tokens) != len(out_tokens):
        merged_tags = fix_bio_sequence(merged_tags, id2label, label2id)

    return merged_tokens, merged_tags


# ---------------------------------------------------------------------------
# ASR noise
# ---------------------------------------------------------------------------

def apply_asr_noise(
    tokens: List[str],
    tags: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    p: float = 0.3,
    seed: int = None,
) -> Tuple[List[str], List[int]]:
    """
    Simulate Automatic Speech Recognition (ASR) transcription noise.

    Always applied:
      1. Lowercase all tokens.
      2. Remove purely punctuation tokens (and their tags); fix BIO.

    With probability *p* per token:
      3. Substitute word via PHONETIC_MAP.

    For O-tagged function words, with probability p*0.4:
      4. Delete token and tag; fix BIO.

    Parameters
    ----------
    tokens    : word-level token list
    tags      : corresponding integer NER tag list
    id2label  : id -> label string mapping
    label2id  : label string -> id mapping
    p         : per-token noise probability (default 0.3)
    seed      : optional random seed for reproducibility

    Returns
    -------
    (noisy_tokens, noisy_tags)
    """
    if seed is not None:
        random.seed(seed)

    if not tokens:
        return tokens, tags

    out_tokens = list(tokens)
    out_tags = list(tags)

    # Step 1: lowercase all tokens
    out_tokens = [tok.lower() for tok in out_tokens]

    # Step 2: remove purely punctuation tokens
    punct_set = set(string.punctuation)
    filtered_tokens: List[str] = []
    filtered_tags: List[int] = []
    removed_punct = False

    for tok, tag in zip(out_tokens, out_tags):
        if tok and all(ch in punct_set for ch in tok):
            removed_punct = True
        else:
            filtered_tokens.append(tok)
            filtered_tags.append(tag)

    if removed_punct:
        filtered_tags = fix_bio_sequence(filtered_tags, id2label, label2id)

    out_tokens = filtered_tokens
    out_tags = filtered_tags

    # Steps 3 & 4: phonetic substitution and function-word deletion
    final_tokens: List[str] = []
    final_tags: List[int] = []
    deletion_happened = False

    for tok, tag in zip(out_tokens, out_tags):
        label = id2label[tag]

        # Step 3: phonetic substitution (independent draw)
        if random.random() < p:
            substituted = PHONETIC_MAP.get(tok.lower())
            if substituted is not None:
                tok = substituted

        # Step 4: function-word deletion (independent draw, O-tagged only)
        if label == 'O' and tok.lower() in FUNCTION_WORDS:
            if random.random() < p * 0.4:
                deletion_happened = True
                continue  # skip appending → delete token

        final_tokens.append(tok)
        final_tags.append(tag)

    if deletion_happened:
        final_tags = fix_bio_sequence(final_tags, id2label, label2id)

    return final_tokens, final_tags


# ---------------------------------------------------------------------------
# Social media noise
# ---------------------------------------------------------------------------

def apply_social_noise(
    tokens: List[str],
    tags: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    p: float = 0.3,
    seed: int = None,
) -> Tuple[List[str], List[int]]:
    """
    Simulate social-media style text noise.

    For each token with probability *p* (operations applied independently):
      1. Keyboard typo: replace one random character with a QWERTY neighbor.
      2. With prob 0.3 (and O-tagged): repeat last character 2-3 times.
      3. With prob 0.4: randomly uppercase ~30% of characters.

    For O-tagged tokens only (independent draws):
      4. With prob p*0.5: replace with SOCIAL_ABBREV entry (lowercase match).
      5. With prob p*0.3: remove vowels from interior (keep first & last char),
         only if len(token) > 4.

    Parameters
    ----------
    tokens    : word-level token list
    tags      : corresponding integer NER tag list
    id2label  : id -> label string mapping
    label2id  : label string -> id mapping
    p         : per-token noise probability (default 0.3)
    seed      : optional random seed for reproducibility

    Returns
    -------
    (noisy_tokens, noisy_tags) — same length as input (no deletions here)
    """
    if seed is not None:
        random.seed(seed)

    if not tokens:
        return tokens, tags

    out_tokens = list(tokens)
    out_tags = list(tags)
    vowels = set('aeiouAEIOU')

    for i in range(len(out_tokens)):
        tok = out_tokens[i]
        tag = out_tags[i]
        label = id2label[tag]
        is_o = (label == 'O')

        if random.random() < p:
            # Step 1: keyboard typo — replace one random char with QWERTY neighbor
            candidate_indices = [
                j for j, ch in enumerate(tok)
                if ch.lower() in QWERTY_NEIGHBORS
            ]
            if candidate_indices:
                j = random.choice(candidate_indices)
                ch = tok[j]
                neighbors = QWERTY_NEIGHBORS.get(ch.lower(), [])
                if neighbors:
                    replacement = random.choice(neighbors)
                    tok = tok[:j] + replacement + tok[j + 1:]

            # Step 2: repeat last character 2-3 times (O-tagged only)
            if random.random() < 0.3 and is_o and len(tok) >= 1:
                repeat_count = random.randint(2, 3)
                tok = tok + tok[-1] * repeat_count

            # Step 3: random CaPs (~30% of chars uppercased)
            if random.random() < 0.4:
                tok = ''.join(
                    ch.upper() if random.random() < 0.3 else ch
                    for ch in tok
                )

        # Steps 4 & 5: O-tagged tokens only (independent of the p gate above)
        if is_o:
            # Step 4: social abbreviation substitution
            if random.random() < p * 0.5:
                abbrev = SOCIAL_ABBREV.get(tok.lower())
                if abbrev is not None:
                    tok = abbrev

            # Step 5: remove interior vowels (keep first and last char)
            if random.random() < p * 0.3 and len(tok) > 4:
                first = tok[0]
                last = tok[-1]
                middle = tok[1:-1]
                middle_no_vowels = ''.join(ch for ch in middle if ch not in vowels)
                tok = first + middle_no_vowels + last

        out_tokens[i] = tok

    return out_tokens, out_tags
