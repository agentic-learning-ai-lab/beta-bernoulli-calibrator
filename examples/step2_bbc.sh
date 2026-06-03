#!/usr/bin/env bash
# BBC sweep for the main table:
#   4 inputs x 2 variants (binary, binary+human) x 2 lr {1e-6, 5e-6}
#   x 2 LoRA rank {128, 256} x 3 seeds {0, 42, 123}  = 96 runs.
# Each run is independent; resume by skipping completed output dirs.
set -e

INPUTS=(claude-4-sonnet llama3-70b qwen3-32b qwen3-8b)
LRS=(1e-6 5e-6)
RANKS=(128 256)
SEEDS=(0 42 123)
HUMAN_COEFFS=(0 1)

for INPUT in "${INPUTS[@]}"; do
  for HC in "${HUMAN_COEFFS[@]}"; do
    for LR in "${LRS[@]}"; do
      for RANK in "${RANKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          OUT="./results/default/bbc/out_llama3_2-1b/in_$INPUT/binary-1-human-$HC-rank-$RANK-lr-$LR-seed-$SEED"
          if [ -f "$OUT/test_dataset_with_preds.json" ]; then
            echo "skip: $OUT (already done)"
            continue
          fi
          python ./bbc/train.py \
            --train_path "./results/default/verbalized/$INPUT/train.json" \
            --val_path   "./results/default/verbalized/$INPUT/val.json" \
            --test_path  "./results/default/verbalized/$INPUT/test.json" \
            --net meta-llama/Llama-3.2-1B \
            --epochs 15 \
            --binary_coeff 1 \
            --human_coeff "$HC" \
            --no_reasoning \
            --lr "$LR" \
            --lora_rank "$RANK" \
            --n_models 5 \
            --print_interval 300 \
            --seed "$SEED" \
            --output "$OUT"
        done
      done
    done
  done
done
