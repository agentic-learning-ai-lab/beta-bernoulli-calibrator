#!/usr/bin/env bash
# Produces:
#   - Main table (stdout)
#   - analysis/figures/reliability_{bbc,baseline}.pdf
#   - analysis/figures/uncertainty_{bbc,sampling,verbalized}.pdf
set -e

INPUTS=(claude-4-sonnet llama3-70b qwen3-32b qwen3-8b)

# (a) Verbalized confidence: generate raw + process (uncertainty_plot.py --mode verbalized).
for INPUT in "${INPUTS[@]}"; do
  python verbalized/generate.py \
    --mode w_conf \
    --model_name "$INPUT" \
    --input_path ./data/default/test.json \
    --output_path ./results/default/verbalized_w_conf
  python verbalized/process.py \
    --input  "./results/default/verbalized_w_conf/$INPUT/test_raw.json" \
    --output "./results/default/verbalized_w_conf/$INPUT/test.json"
done

# (b) Table + figures.
python analysis/main_table.py
python analysis/reliability_diagram.py --mode bbc
python analysis/reliability_diagram.py --mode baseline
python analysis/uncertainty_plot.py --mode bbc
python analysis/uncertainty_plot.py --mode sampling
python analysis/uncertainty_plot.py --mode verbalized
