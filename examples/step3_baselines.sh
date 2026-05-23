#!/usr/bin/env bash
# Baselines: Ensemble, P(True), Platt, Isotonic, and fine-tuned forecasters.
set -e

INPUTS=(claude-4-sonnet llama3-70b qwen3-32b qwen3-8b)
FINETUNED_INPUTS=(openforecaster-8b future-as-label-32b)

for INPUT in "${INPUTS[@]}"; do
  echo "================  $INPUT  ================"

  # (a) Ensemble baseline (n=3 for Claude, n=10 otherwise).
  N=10; [ "$INPUT" = "claude-4-sonnet" ] && N=3
  python verbalized/generate.py \
    --mode ensemble \
    --model_name "$INPUT" \
    --input_path ./data/default/test.json \
    --output_path ./results/default/ensemble \
    --n_samples "$N" \
    --temperature 1.0
  python verbalized/process.py \
    --input  "./results/default/ensemble/$INPUT/test_raw.json" \
    --output "./results/default/ensemble/$INPUT/test.json"

  # (b) P(True) — whitebox; skip Claude (no logit access).
  if [ "$INPUT" != "claude-4-sonnet" ]; then
    python baselines/ptrue.py \
      --model_name "$INPUT" \
      --mode rationale \
      --input_path ./data/default/test.json \
      --output_path ./results/default/ptrue
  fi

  # (c) Platt + Isotonic, fit on train+val, applied to test.
  python ./baselines/platt.py \
    --train_data "./results/default/verbalized/$INPUT/train.json" \
    --val_data   "./results/default/verbalized/$INPUT/val.json" \
    --target_data "./results/default/verbalized/$INPUT/test.json" \
    --output_dir "./results/default/platt/$INPUT" \
    --method platt

  python ./baselines/isotonic.py \
    --train_data "./results/default/verbalized/$INPUT/train.json" \
    --val_data   "./results/default/verbalized/$INPUT/val.json" \
    --target_data "./results/default/verbalized/$INPUT/test.json" \
    --output_dir "./results/default/isotonic/$INPUT"
done

# (d) Fine-tuned forecaster baselines — verbalized forecast on test set.
#     OpenForecaster-8B:        https://huggingface.co/nikhilchandak/OpenForecaster-8B
#     Future-as-a-Label-32B:    https://huggingface.co/LightningRodLabs/future-as-label-paper-step160
for INPUT in "${FINETUNED_INPUTS[@]}"; do
  echo "================  $INPUT  ================"
  python verbalized/generate.py \
    --model_name "$INPUT" \
    --input_path ./data/default/test.json \
    --output_path ./results/default/verbalized
  python verbalized/process.py \
    --input  "./results/default/verbalized/$INPUT/test_raw.json" \
    --output "./results/default/verbalized/$INPUT/test.json"
done
