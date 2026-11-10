#!/bin/bash
# Zero-shot bake-off on the validation split (never the test split): 2 models x 2 prompts.
set -e
cd "$(dirname "$0")/.."
for M in LFM2.5-350M granite-4.0-h-350m; do
  for P in compact schema; do
    .venv/bin/python scripts/eval_hf.py --model models/$M --split val --prompt $P --name zeroshot_val_${M}_${P} --batch 16
  done
done
