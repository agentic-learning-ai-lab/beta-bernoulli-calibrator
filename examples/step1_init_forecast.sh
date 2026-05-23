#!/usr/bin/env bash
# Initial verbalized forecasts for the 4 main input LLMs (train/val/test).
set -e

INPUTS=(claude-4-sonnet llama3-70b qwen3-32b qwen3-8b)

for INPUT in "${INPUTS[@]}"; do
  echo "================  $INPUT  ================"
  for split in train val test; do
    python verbalized/generate.py \
      --model_name "$INPUT" \
      --input_path "./data/default/${split}.json" \
      --output_path ./results/default/verbalized
    python verbalized/process.py \
      --input  "./results/default/verbalized/$INPUT/${split}_raw.json" \
      --output "./results/default/verbalized/$INPUT/${split}.json"
  done
done
