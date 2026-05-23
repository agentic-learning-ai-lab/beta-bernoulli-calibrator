import numpy as np
from sklearn.isotonic import IsotonicRegression
import pandas as pd
from utils import get_metrics
import argparse
import os

"""
Isotonic regression baseline.
Reads `prob` (parsed initial verbalized forecast) and writes calibrated `pred`.

Example (run from the project root):
python ./baselines/isotonic.py \
  --train_data ./results/default/verbalized/qwen3-8b/train.json \
  --val_data   ./results/default/verbalized/qwen3-8b/val.json \
  --target_data ./results/default/verbalized/qwen3-8b/test.json \
  --output_dir ./results/default/isotonic/qwen3-8b
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit calibrator on val set and apply to target set.")
    ap.add_argument("--train_data", required=False, default=None)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--out_col", default="pred")
    args = ap.parse_args()


    # --- load val (fit) ---
    if args.train_data is not None:
        train_df = pd.read_json(args.train_data, convert_dates=False)
        val_df = pd.read_json(args.val_data, convert_dates=False)
        val_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        val_df = pd.read_json(args.val_data, convert_dates=False)
    p_val = val_df["prob"].astype(float).to_numpy()
    y_val = np.array([1 if i == 'yes' else 0 for i in val_df['resolution']])

    # --- fit isotonic ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)


    # --- load test (apply) ---
    test_df = pd.read_json(args.target_data, convert_dates=False)
    p_test = test_df["prob"].astype(float).to_numpy()
    y_test = np.array([1 if i == 'yes' else 0 for i in test_df['resolution']])

    p_test_cal = np.full_like(p_test, np.nan, dtype=float)
    p_test_cal = iso.transform(p_test)
    test_df[args.out_col] = p_test_cal
    print("---- Before ----")
    get_metrics(test_df, "prob")

    print("---- After ----")
    get_metrics(test_df, args.out_col)


    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.target_data).split(".")[0]
    save_path = os.path.join(args.output_dir, f"{file_name}.json")
    test_df.to_json(save_path, orient="records")
    print(f"Wrote {args.out_col} to {save_path}")
