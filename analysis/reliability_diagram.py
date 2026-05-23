"""
Reliability diagram.

  --mode bbc        per-LLM curve = mean ± 1 std across the top-5 BBC runs
                    (by val_brier), each computed on `pred` from
                    ./results/default/bbc/out_<OUT>/in_<INPUT>/<run>/test_dataset_with_preds.json
  --mode baseline   per-LLM curve from `prob` in
                    ./results/default/verbalized/<INPUT>/test.json (no band, deterministic)

Usage:
  python analysis/reliability_diagram.py  # default mode = bbc
  python analysis/reliability_diagram.py --mode baseline
  python analysis/reliability_diagram.py --inputs claude-4-sonnet qwen3-32b
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


OUT_MODEL = "llama3_2-1b"
BBC_BASE = "./results/default/bbc"
VALID_LR = {1e-6, 5e-6}
VALID_RANK = {128, 256}
VALID_EPOCHS = {15}
VALID_SEEDS = {0, 42, 123}
K_TOP = 5

DEFAULT_INPUTS = ["claude-4-sonnet", "llama3-70b", "qwen3-32b", "qwen3-8b"]

MODEL_PRETTY = {
    "claude-4-sonnet": "Claude-Sonnet-4",
    "llama3-70b":      "Llama-3.3-70B-Instruct",
    "qwen3-32b":       "Qwen3-32B",
    "qwen3-8b":        "Qwen3-8B",
}


# -- run selection -----------------------------------------------------------

def _topk_paths(input_model: str, k: int = K_TOP) -> List[str]:
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    if not os.path.isdir(base):
        return []
    items = []
    for d in sorted(os.listdir(base)):
        cfg_path = f"{base}/{d}/config.json"
        hist_path = f"{base}/{d}/history.json"
        if not (os.path.exists(cfg_path) and os.path.exists(hist_path)):
            continue
        cfg = json.load(open(cfg_path))
        if (cfg.get("binary_coeff") == 1.0
                and cfg.get("human_coeff") == 1.0
                and cfg.get("learning_rate") in VALID_LR
                and cfg.get("lora_rank") in VALID_RANK
                and cfg.get("epochs") in VALID_EPOCHS
                and cfg.get("seed", 42) in VALID_SEEDS):
            v = float(pd.read_json(hist_path, convert_dates=False)["val_brier"].min())
            items.append((v, d))
    items.sort()
    return [d for _, d in items[:k]]


# -- per-run reliability -----------------------------------------------------

def _reliability(preds: np.ndarray, actuals: np.ndarray,
                 bin_edges: np.ndarray, min_bin_size: int) -> np.ndarray:
    """Return per-bin empirical positive rate (NaN where the bin is too sparse)."""
    n_bins = len(bin_edges) - 1
    bin_ids = np.digitize(preds, bin_edges[1:-1], right=True)
    prob_true = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() >= min_bin_size:
            prob_true[i] = actuals[mask].mean()
    return prob_true


def _resolution_labels(df: pd.DataFrame) -> np.ndarray:
    return np.array([1 if str(r).lower() == "yes" else 0 for r in df["resolution"]], dtype=int)


def _bbc_runs(input_model: str, bin_edges: np.ndarray, min_bin_size: int) -> Optional[np.ndarray]:
    """Stack per-run reliability curves for the top-5 BBC runs. Shape [n_runs, n_bins]."""
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    runs = _topk_paths(input_model)
    curves = []
    for d in runs:
        fp = f"{base}/{d}/test_dataset_with_preds.json"
        if not os.path.exists(fp):
            continue
        df = pd.read_json(fp, convert_dates=False)
        y = _resolution_labels(df)
        p = np.clip(df["pred"].astype(float).values, 0, 1)
        curves.append(_reliability(p, y, bin_edges, min_bin_size))
    return np.asarray(curves) if curves else None


def _baseline_curve(input_model: str, bin_edges: np.ndarray, min_bin_size: int) -> Optional[np.ndarray]:
    fp = f"./results/default/verbalized/{input_model}/test.json"
    if not os.path.exists(fp):
        return None
    df = pd.read_json(fp, convert_dates=False)
    y = _resolution_labels(df)
    p = np.clip(df["prob"].astype(float).values, 0, 1)
    return _reliability(p, y, bin_edges, min_bin_size)


# -- plot --------------------------------------------------------------------

MARKERS = ["o", "s", "^", "*", "D", "P", "v", "X"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bbc", "baseline"], default="bbc")
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--min_bin_size", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="Output figure path. Default: analysis/figures/reliability_<mode>.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 2.8,
        "lines.markersize": 10,
    })

    bin_edges = np.linspace(0, 1, args.n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2.5, label="Perfectly calibrated")

    for k_idx, input_model in enumerate(args.inputs):
        if args.mode == "bbc":
            R = _bbc_runs(input_model, bin_edges, args.min_bin_size)
            if R is None:
                continue
            n_valid = np.sum(~np.isnan(R), axis=0)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(R, axis=0)
                std  = np.nanstd(R, axis=0, ddof=1)
            std = np.where(n_valid >= 2, std, 0.0)
        else:  # baseline
            mean = _baseline_curve(input_model, bin_edges, args.min_bin_size)
            if mean is None:
                continue
            std = None

        valid = ~np.isnan(mean)
        color = colors[k_idx % len(colors)]
        ax.plot(bin_centers[valid], mean[valid],
                marker=MARKERS[k_idx % len(MARKERS)], markersize=11,
                linewidth=3.0, alpha=0.9,
                color=color, label=MODEL_PRETTY.get(input_model, input_model))
        if std is not None:
            ax.fill_between(bin_centers[valid],
                            (mean - std)[valid], (mean + std)[valid],
                            alpha=0.18, color=color, linewidth=0)

    ax.set_xlabel("Mean Forecast Probability")
    ax.set_ylabel("Mean Actual Outcome")
    ax.set_title("Reliability Diagram: Beta-Bernoulli Calibrator"
                 if args.mode == "bbc" else "Reliability Diagram: Verbalized Baseline")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    leg = ax.legend(loc="upper left", fontsize=20, handlelength=3.2,
                    handletextpad=0.7, borderpad=0.7, labelspacing=0.5,
                    frameon=True, framealpha=0.92)
    for lh in leg.legend_handles:
        try:
            lh.set_linewidth(4.5)
        except AttributeError:
            pass

    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    out = args.out or f"analysis/figures/reliability_{args.mode}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
