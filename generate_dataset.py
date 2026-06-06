"""
generate_dataset.py

Generates the compositional staircase dataset for the LoRA sandbox experiment.
Pure Python stdlib — no ML dependencies. Run locally before uploading to Colab.

Output: data/level1/  (train.jsonl, val_id.jsonl, val_ood.jsonl)
"""

import json
import random
import pathlib


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = pathlib.Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pass  # dataset generation logic goes here


if __name__ == "__main__":
    main()
