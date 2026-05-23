"""
Uncertainty vs Brier-loss plot.

  --mode verbalized   u = 1 - confidence          (results/default/verbalized_w_conf/<INPUT>/test.json)
  --mode sampling     u = Var_k(p)                (results/default/ensemble/<INPUT>/test.json)
  --mode bbc          u = mixture variance        (top-5 BBC runs averaged per question)

Usage:
  python analysis/uncertainty_plot.py --mode bbc
  python analysis/uncertainty_plot.py --mode sampling --inputs claude-4-sonnet qwen3-32b
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


OUT_MODEL = "llama3_2-1b"
BBC_BASE = "./results/default/bbc"
VERB_BASE = "./results/default"

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



# -- BBC selection -----------------------------------------------------------

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


# -- p, u extractors ---------------------------------------------------------

def bbc_p_and_u(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Mixture-of-Beta mean and variance per row, from `alpha`, `beta`, `weight`."""
    eps = 1e-12
    A = np.vstack(df["alpha"].to_numpy())
    B = np.vstack(df["beta"].to_numpy())
    W = np.vstack(df["weight"].to_numpy())
    W = W / np.clip(W.sum(axis=1, keepdims=True), eps, None)
    S = A + B
    mk = A / np.clip(S, eps, None)
    vk = (A * B) / (np.clip(S, eps, None) ** 2 * np.clip(S + 1.0, eps, None))
    p = np.sum(W * mk, axis=1)
    second_moment = np.sum(W * (vk + mk ** 2), axis=1)
    u = np.clip(second_moment - p ** 2, 0.0, None)
    return p, u


def human_p_and_u(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Human histogram mean and variance per row."""
    eps = 1e-12
    H = np.vstack(df["forecast_histogram"].to_numpy()).astype(np.float64)
    H = H / np.clip(H.sum(axis=1, keepdims=True), eps, None)
    K = H.shape[1]
    centers = (np.arange(K) + 0.5) / K
    mu = H @ centers
    second_moment = H @ (centers ** 2)
    u = np.clip(second_moment - mu ** 2, 0.0, None)
    return mu, u


# -- per-mode (p, u) loaders --------------------------------------------------

def load_verbalized(input_model: str) -> Optional[pd.DataFrame]:
    """Reads the processed w_conf file (has `prob` and `confidence`)."""
    fp = f"{VERB_BASE}/verbalized_w_conf/{input_model}/test.json"
    if not os.path.exists(fp):
        return None
    df = pd.read_json(fp, convert_dates=False)
    df = df.assign(p=df["prob"], u=1 - df["confidence"])
    return df[["p", "u", "resolution"]]


def load_sampling(input_model: str) -> Optional[pd.DataFrame]:
    """Reads the processed ensemble file (has `prob_1`..`prob_n`)."""
    fp = f"{VERB_BASE}/ensemble/{input_model}/test.json"
    if not os.path.exists(fp):
        return None
    df = pd.read_json(fp, convert_dates=False)
    prob_cols = sorted([c for c in df.columns if re.fullmatch(r"prob_\d+", c)],
                       key=lambda s: int(s.split("_")[1]))
    if not prob_cols:
        return None
    P = np.clip(df[prob_cols].to_numpy(dtype=float), 0.0, 1.0)
    df = df.assign(p=P.mean(axis=1), u=P.var(axis=1, ddof=0))
    return df[["p", "u", "resolution"]]


def load_bbc(input_model: str) -> Optional[pd.DataFrame]:
    paths = _topk_paths(input_model)
    base = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}"
    ps, us, ref = [], [], None
    for d in paths:
        fp = f"{base}/{d}/test_dataset_with_preds.json"
        if not os.path.exists(fp):
            continue
        df_run = pd.read_json(fp, convert_dates=False)
        p, u = bbc_p_and_u(df_run)
        ps.append(p); us.append(u)
        if ref is None:
            ref = df_run[["resolution"]].copy()
    if not ps or ref is None:
        return None
    out = ref.copy()
    out["p"] = np.mean(np.vstack(ps), axis=0)
    out["u"] = np.mean(np.vstack(us), axis=0)
    return out[["p", "u", "resolution"]]


def load_human_reference(input_model: str = "claude-4-sonnet") -> Optional[pd.DataFrame]:
    """Use any one BBC test file to read forecast_histogram and derive (p, u) for humans."""
    paths = _topk_paths(input_model, k=1)
    if not paths:
        return None
    fp = f"{BBC_BASE}/out_{OUT_MODEL}/in_{input_model}/{paths[0]}/test_dataset_with_preds.json"
    if not os.path.exists(fp):
        return None
    d = pd.read_json(fp, convert_dates=False)
    if "forecast_histogram" not in d.columns:
        return None
    p, u = human_p_and_u(d)
    return pd.DataFrame({"p": p, "u": u, "resolution": d["resolution"]})


# -- rolling-window plot -----------------------------------------------------

def plot_loss_vs_u(ax, df: pd.DataFrame, label: str, window: int = 300,
                   color: Optional[str] = None, band: Optional[str] = "se"):
    df = df.copy()
    df["y"] = (df["resolution"].astype(str).str.lower() == "yes").astype(float)
    df["loss2"] = (df["y"] - df["p"]) ** 2
    df = df.sort_values("u").reset_index(drop=True)

    minp = max(window // 5, 1)
    roll = df["loss2"].rolling(window, center=True, min_periods=minp)
    roll_u = df["u"].rolling(window, center=True, min_periods=minp).mean()
    roll_mean = roll.mean()
    roll_std = roll.std()
    roll_n = roll.count()

    line, = ax.plot(roll_u, roll_mean, label=label, color=color)
    if band == "se":
        half = roll_std / np.sqrt(roll_n)
    elif band == "std":
        half = roll_std
    else:
        return
    ax.fill_between(roll_u, roll_mean - half, roll_mean + half,
                    alpha=0.18, color=line.get_color(), linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verbalized", "sampling", "bbc"], required=True)
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="Output figure path. Defaults to analysis/figures/uncertainty_<mode>.pdf")
    args = ap.parse_args()

    loader = {"verbalized": load_verbalized, "sampling": load_sampling, "bbc": load_bbc}[args.mode]
    show_human = args.mode in {"sampling", "bbc"}

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    if show_human:
        human = load_human_reference()
        if human is not None:
            plot_loss_vs_u(ax, human, "Human Forecast",
                           window=args.window, color="black", band=None)

    for input_model in args.inputs:
        df = loader(input_model)
        if df is None:
            print(f"  [MISSING] {input_model} in mode={args.mode}; skipping")
            continue
        plot_loss_vs_u(ax, df, MODEL_PRETTY.get(input_model, input_model),
                       window=args.window)

    xlabel = {
        "verbalized": r"Uncertainty $u = 1 - \mathrm{confidence}$",
        "sampling":   r"Uncertainty $u = \mathrm{Var}(p)$",
        "bbc":        r"Uncertainty $u = \mathrm{Var}(p)$",
    }[args.mode]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Brier loss")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.out or f"analysis/figures/uncertainty_{args.mode}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
