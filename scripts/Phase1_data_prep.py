#!/usr/bin/env python3
"""
Phase 1: Data Preparation for arthur v2.0
Expand training data from 1.8M to 50M+ tokens.

Tasks:
1. Verify WikiText-103 is downloaded locally
2. Download ArXiv papers metadata (code papers)
3. Prepare balanced dataset (70% knowledge, 10% code, 15% reasoning, 5% math)
4. Train BPE tokenizer
5. Create training shards

Run: python3 Phase1_data_prep.py
"""

import os
import sys
import json
from pathlib import Path

ARTHUR_ROOT = Path(__file__).parent
DATA_DIR = ARTHUR_ROOT / "data"
MODELS_DIR = ARTHUR_ROOT / "models"

def check_wikitext103():
    """Verify WikiText-103 is present."""
    wikitext_path = DATA_DIR / "wikitext-103-raw"
    if wikitext_path.exists():
        print(f"✓ WikiText-103 found at {wikitext_path}")
        files = list(wikitext_path.glob("**/train*"))
        size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"  Train files: {len(files)}, Size: {size_mb:.0f}MB")
        return True
    else:
        print(f"✗ WikiText-103 not found. Download with:")
        print(f"  wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip")
        print(f"  unzip wikitext-103-raw-v1.zip -d {DATA_DIR}")
        return False

def check_datasets():
    """Check which datasets are available."""
    datasets = {
        'wikitext': DATA_DIR / "wikitext-103-raw",
        'math': DATA_DIR / "MATH-500",
        'gsm8k': DATA_DIR / "gsm8k",
        'arxiv': DATA_DIR / "arxiv-papers",
    }
    
    print("\nDataset Status:")
    for name, path in datasets.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    
    return datasets

def create_data_plan():
    """Create data expansion plan."""
    plan = {
        'current': {
            'size_tokens': 1.8e6,
            'components': {
                'math': 900000,
                'wikipedia': 900000,
            }
        },
        'target': {
            'size_tokens': 50e6,
            'components': {
                'knowledge': 35e6,  # 70% WikiText-103, ArXiv abstracts
                'code': 5e6,        # 10% GitHub (public Python/JS)
                'reasoning': 7.5e6, # 15% Math problems, logic puzzles
                'math': 2.5e6,      # 5% MATH-500, GSM8K
            }
        }
    }
    
    print("\nData Expansion Plan:")
    print(f"  Current: {plan['current']['size_tokens']/1e6:.1f}M tokens")
    print(f"  Target: {plan['target']['size_tokens']/1e6:.1f}M tokens")
    print(f"  Scaling: {plan['target']['size_tokens'] / plan['current']['size_tokens']:.0f}x")
    
    print("\nTarget Composition:")
    for component, size in plan['target']['components'].items():
        pct = 100 * size / plan['target']['size_tokens']
        print(f"  {component}: {size/1e6:.1f}M tokens ({pct:.0f}%)")
    
    return plan

def main():
    print("=== ARTHUR V2.0: PHASE 1 DATA PREPARATION ===\n")
    
    # Check existing data
    has_wikitext = check_wikitext103()
    datasets = check_datasets()
    plan = create_data_plan()
    
    # Next steps
    print("\nNext Steps:")
    print("1. Download WikiText-103 (1.5M tokens)")
    print("   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip")
    print("2. Download ArXiv papers (arxiv-metadata-oai-snapshot.json)")
    print("   See: https://www.kaggle.com/datasets/Cornell-University/arxiv")
    print("3. Clone MATH-500 dataset")
    print("   git clone https://github.com/hendrycks/math.git")
    print("4. Download GSM8K")
    print("   wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl")
    print("5. Once data is collected, run: python3 Phase1_tokenizer_train.py")
    
    print("\nEstimated Timeline:")
    print("  - Data download: 2-3 hours (parallelize)")
    print("  - BPE training: 30 mins (single pass)")
    print("  - Dataset creation: 1 hour")
    print("  - Total Phase 1: 4-5 hours")

if __name__ == '__main__':
    main()
