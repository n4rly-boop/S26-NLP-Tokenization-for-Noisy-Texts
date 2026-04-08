"""
Download CoNLL-2003 dataset from HuggingFace and save to data/raw/.

Usage:
    python data/download_data.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "raw")


def download_conll2003(output_dir: str = OUTPUT_DIR) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading CoNLL-2003 from HuggingFace...")
    dataset = load_dataset("conll2003", trust_remote_code=True)

    # Save each split as JSONL
    for split in ["train", "validation", "test"]:
        output_path = os.path.join(output_dir, f"conll2003_{split}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset[split]:
                f.write(json.dumps(example) + "\n")
        print(f"  [{split:10s}] {len(dataset[split]):5d} examples → {output_path}")

    # Save label names for NER tags
    label_names = dataset["train"].features["ner_tags"].feature.names
    label_path = os.path.join(output_dir, "ner_label_names.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    print(f"\nNER labels ({len(label_names)}): {label_names}")
    print(f"Label map saved → {label_path}")
    print("\nDone. Data ready in data/raw/")


if __name__ == "__main__":
    download_conll2003()
