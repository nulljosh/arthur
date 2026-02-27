#!/usr/bin/env python3
"""Download WikiText-103 dataset from HuggingFace."""
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "wikipedia"

def main():
    from datasets import load_dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    for split in ["train", "validation", "test"]:
        out = DATA_DIR / f"{split}.txt"
        texts = dataset[split]["text"]
        cleaned = [t for t in texts if t.strip() and not t.strip().startswith("=")]
        with open(out, "w") as f:
            f.write("\n".join(cleaned))
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"  {split}: {len(cleaned):,} lines, {size_mb:.1f} MB -> {out}")
    print("Done.")

if __name__ == "__main__":
    main()
