"""
Reproduce the main table (Verbalized / Ensemble / Platt / Isotonic / P(True) /
fine-tuned forecaster / BBC) for the input LLMs in the paper.

Reads (run from project root):
  Verbalized           ./results/default/verbalized/<INPUT>/test.json     (`prob`)
  Ensemble             ./results/default/ensemble/<INPUT>/test.json       (`prob` = mean of K samples)
  Platt Scaling        ./results/default/platt/<INPUT>/test.json          (`pred`)
  Isotonic Regression  ./results/default/isotonic/<INPUT>/test.json       (`pred`)
  P(True)              ./results/default/ptrue/<INPUT>/test.json          (`p_yes`)
  BBC (binary)         ./results/default/bbc/out_<OUT>/in_<INPUT>/<run>/  (top-5 by val_brier)
  BBC (binary+human)   ./results/default/bbc/out_<OUT>/in_<INPUT>/<run>/  (top-5 by val_brier)

The BBC sweep is the paper grid: 2 lr {1e-6, 5e-6} × 2 LoRA rank {128, 256} × 3 seeds {0, 42, 123}.

Usage:
  python analysis/main_table.py
  python analysis/main_table.py --inputs claude-4-sonnet qwen3-32b
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_metrics, get_kl  # noqa: E402

# -- defaults ----------------------------------------------------------------

OUT_MODEL = "llama3_2-1b"
BBC_BASE = "./results/default/bbc"
VALID_LR = {1e-6, 5e-6}
VALID_RANK = {128, 256}
VALID_EPOCHS = {15}
VALID_SEEDS = {0, 42, 123}
K_TOP = 5

# Per-input LLM ensemble sample count (paper defaults).
ENSEMBLE_N = {
    "claude-4-sonnet": 3,
    "llama3-70b": 10,
    "qwen3-32b": 10,
    "qwen3-8b": 10,
}

DEFAULT_INPUTS = ["claude-4-sonnet", "llama3-70b", "qwen3-32b", "qwen3-8b"]

# Fine-tuned forecaster row attached to a specific input LLM in the table.
FINETUNED_FOR = {
    "qwen3-32b": "future-as-label-32b",
    "qwen3-8b":  "openforecaster-8b",
}

# Whitebox-only baselines (no API access to logits for Claude).
PTRUE_ELIGIBLE = {"llama3-70b", "qwen3-32b", "qwen3-8b"}


# -- helpers -----------------------------------------------------------------

def _safe_read_json(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_json(path, convert_dates=False)


def _metrics_from_prob(df: pd.DataFrame, prob_col: str) -> Optional[dict]:
    if df is None or prob_col not in df.columns:
        return None
    return get_metrics(df, prob_col, print_metrics=False)


def _row(name: str, m: Optional[dict], kl: Optional[float] = None) -> dict:
    if m is None:
        return {"method": name, "brier": None, "acc": None, "auc": None, "ece": None, "kl": kl}
    return {
        "method": name,
        "brier": m["brier"], "acc": m["accuracy"], "auc": m["auc"], "ece": m["ece"],
        "kl": kl,
    }


def _bbc_run_dirs(input_model: str, human_coeff: int) -> List[str]:
    """Walk results dir, return run subdirs whose config matches the paper sweep."""
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    if not os.path.isdir(base):
        return []
    keep = []
    for d in sorted(os.listdir(base)):
        cfg_path = f"{base}/{d}/config.json"
        hist_path = f"{base}/{d}/history.json"
        if not (os.path.exists(cfg_path) and os.path.exists(hist_path)):
            continue
        cfg = json.load(open(cfg_path))
        if (cfg.get("binary_coeff") == 1.0
                and cfg.get("human_coeff") == float(human_coeff)
                and cfg.get("learning_rate") in VALID_LR
                and cfg.get("lora_rank") in VALID_RANK
                and cfg.get("epochs") in VALID_EPOCHS
                and cfg.get("seed", 42) in VALID_SEEDS):
            keep.append(d)
    return keep


def _topk_by_val(input_model: str, human_coeff: int, k: int = K_TOP) -> List[str]:
    runs = _bbc_run_dirs(input_model, human_coeff)
    if not runs:
        return []
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    scored = []
    for d in runs:
        h = pd.read_json(f"{base}/{d}/history.json", convert_dates=False)
        scored.append((float(h["val_brier"].min()), d))
    scored.sort()
    return [d for _, d in scored[:k]]


def _bbc_metrics(input_model: str, human_coeff: int) -> Tuple[Optional[np.ndarray], int]:
    """Returns (metrics_array, n_runs_used) where metrics_array is shape [k, 5]
    columns = [brier, acc, auc, ece, kl] over the top-K runs.
    """
    runs = _topk_by_val(input_model, human_coeff)
    if not runs:
        return None, 0
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    rows = []
    for d in runs:
        h = pd.read_json(f"{base}/{d}/history.json", convert_dates=False)
        # Best epoch by val_brier; report the test metrics from that epoch.
        idx = h["val_brier"].idxmin()
        bs = round(float(h.at[idx, "test_brier"]), 4)
        ac = round(float(h.at[idx, "test_acc"]), 4)
        au = round(float(h.at[idx, "test_auc"]), 4)
        ec = round(float(h.at[idx, "test_ece"]), 4)
        # KL between predicted mixture and human histogram on test set.
        preds_path = f"{base}/{d}/test_dataset_with_preds.json"
        if os.path.exists(preds_path):
            test_df = pd.read_json(preds_path, convert_dates=False)
            kl = get_kl(test_df)
        else:
            kl = float("nan")
        rows.append([bs, ac, au, ec, kl])
    return np.asarray(rows, dtype=float), len(runs)


# -- baseline readers --------------------------------------------------------

def _verbalized(input_model: str) -> Optional[dict]:
    df = _safe_read_json(f"./results/default/verbalized/{input_model}/test.json")
    return _metrics_from_prob(df, "prob")


def _ensemble(input_model: str) -> Optional[dict]:
    df = _safe_read_json(f"./results/default/ensemble/{input_model}/test.json")
    return _metrics_from_prob(df, "prob")


def _platt(input_model: str) -> Optional[dict]:
    df = _safe_read_json(f"./results/default/platt/{input_model}/test.json")
    return _metrics_from_prob(df, "pred")


def _isotonic(input_model: str) -> Optional[dict]:
    df = _safe_read_json(f"./results/default/isotonic/{input_model}/test.json")
    return _metrics_from_prob(df, "pred")


def _ptrue(input_model: str) -> Optional[dict]:
    if input_model not in PTRUE_ELIGIBLE:
        return None
    df = _safe_read_json(f"./results/default/ptrue/{input_model}/test.json")
    return _metrics_from_prob(df, "p_yes")


# -- formatting --------------------------------------------------------------

def _fmt_mean(v: Optional[float], digits: int = 3) -> str:
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _fmt_meanstd(mean: float, std: Optional[float], digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "  -  "
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({std:.{digits}f})"


def _print_block(input_model: str, rows: List[dict], bbc_rows: List[dict]) -> None:
    print(f"\n=== {input_model} ===")
    header = f"  {'Method':<22} {'Brier':>14} {'Acc':>14} {'AUC':>14} {'ECE':>14} {'KL':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        kl_str = _fmt_mean(r['kl'])
        print(f"  {r['method']:<22} "
              f"{_fmt_mean(r['brier']):>14} {_fmt_mean(r['acc']):>14} "
              f"{_fmt_mean(r['auc']):>14} {_fmt_mean(r['ece']):>14} {kl_str:>14}")
    for r in bbc_rows:
        print(f"  {r['method']:<22} "
              f"{_fmt_meanstd(r['brier'], r.get('brier_std')):>14} "
              f"{_fmt_meanstd(r['acc'], r.get('acc_std')):>14} "
              f"{_fmt_meanstd(r['auc'], r.get('auc_std')):>14} "
              f"{_fmt_meanstd(r['ece'], r.get('ece_std')):>14} "
              f"{_fmt_meanstd(r.get('kl'), r.get('kl_std')):>14}")


def main():
    global OUT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS,
                    help="Input LLMs to include (rows of the table).")
    ap.add_argument("--out_model", default=OUT_MODEL,
                    help="BBC calibrator backbone (short tag, used in result paths).")
    args = ap.parse_args()
    OUT_MODEL = args.out_model

    for input_model in args.inputs:
        rows = []
        rows.append(_row("Verbalized", _verbalized(input_model)))
        rows.append(_row(f"Ensemble (n={ENSEMBLE_N.get(input_model, 10)})", _ensemble(input_model)))
        rows.append(_row("Platt Scaling", _platt(input_model)))
        rows.append(_row("Isotonic Regression", _isotonic(input_model)))
        if input_model in PTRUE_ELIGIBLE:
            rows.append(_row("P(True)", _ptrue(input_model)))
        if input_model in FINETUNED_FOR:
            ft = FINETUNED_FOR[input_model]
            rows.append(_row(ft, _verbalized(ft)))

        bbc_rows = []
        for human_coeff, label in [(0, "BBC (binary only)"), (1, "BBC (binary+human)")]:
            arr, n_runs = _bbc_metrics(input_model, human_coeff)
            if arr is None:
                bbc_rows.append({"method": label, **{c: None for c in
                                                     ["brier", "acc", "auc", "ece", "kl"]}})
                continue
            means = np.nanmean(arr, axis=0)
            stds  = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(means)
            bbc_rows.append({
                "method": f"{label} [top-{n_runs}]",
                "brier": means[0], "acc": means[1], "auc": means[2], "ece": means[3], "kl": means[4],
                "brier_std": stds[0], "acc_std": stds[1], "auc_std": stds[2],
                "ece_std": stds[3], "kl_std": stds[4],
            })

        _print_block(input_model, rows, bbc_rows)


if __name__ == "__main__":
    main()
