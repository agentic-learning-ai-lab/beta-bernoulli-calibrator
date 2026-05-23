#!/usr/bin/env bash
# Run BBC inference on a test split, using a checkpoint produced by examples/step2_bbc.sh.
set -e

python ./bbc/infer.py \
    --model_dir ./results/default/bbc/out_llama3_2-1b/in_claude-4-sonnet/binary-1-human-1-rank-256-lr-1e-6-seed-42 \
    --test_path ./results/default/verbalized/claude-4-sonnet/test.json \
    --output_path ./results/default/bbc/out_llama3_2-1b/in_claude-4-sonnet/binary-1-human-1-rank-256-lr-1e-6-seed-42/test_with_preds.json \
    --batch_size 16
