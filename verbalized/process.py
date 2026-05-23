"""
Parse raw verbalized output from `verbalized/generate.py` into a training- / analysis-ready JSON.

Mode is auto-detected from the input columns:

  default   input has `response` (single string)
            → adds `prob`, `rationale`, and rewrites `response` to the BBC trainer schema
              {"decision_history": [{"reasoning": ..., "parameters": {"Yes": ..., "No": ...}}]}.

  w_conf    input has `response` with a [Confidence:] line
            → same as default, plus `confidence`.

  ensemble  input has `response_1`, ..., `response_n`
            → adds per-sample `prob_k`, `rationale_k`, plus aggregate `prob` (mean) and
              `rationale` (first sample). Keeps the raw `response_k` columns intact.
              Does NOT write the BBC-trainer structured `response` — ensemble is a baseline.

Example (run from the project root):

  # Default / w_conf
  python verbalized/process.py \
    --input ./results/default/verbalized/qwen3-8b/test_raw.json \
    --output ./results/default/verbalized/qwen3-8b/test.json

  # Ensemble
  python verbalized/process.py \
    --input ./results/default/ensemble/qwen3-8b/test_raw.json \
    --output ./results/default/ensemble/qwen3-8b/test.json
"""

import argparse
import os
import re

import numpy as np
import pandas as pd


ANSWER_MARKER = r"(?:\[\s*Answer\s*:\s*\]|\*\*\s*Answer\s*:\s*\*\*|Answer\s*:)"
CONF_MARKER = r"(?:\[\s*Confidence\s*:\s*\]|\*\*\s*Confidence\s*:\s*\*\*|Confidence\s*:)"


def parse_response(text):
    """Parse '[Rationale:] ... [Answer:] <p> [Confidence:] <c>' style output.
    Returns (rationale, prob, confidence_or_None, error_flag)."""
    if not text or not text.strip():
        return "", 0.5, None, True

    pattern = rf"^(?P<rationale>.*?)\s*{ANSWER_MARKER}\s*(?P<rest>.+)$"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return "", 0.5, None, True

    rationale = re.sub(r"^\s*\[\s*Rationale\s*:\s*\]\s*", "",
                       m.group("rationale").strip(), flags=re.IGNORECASE).strip()
    rest = m.group("rest").strip()

    m_num = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\s*%)?", rest)
    if not m_num:
        return rationale, 0.5, None, True
    prob = float(m_num.group(1))
    if m_num.group(2):
        prob /= 100.0
    prob = max(0.0, min(1.0, prob))

    m_conf = re.search(CONF_MARKER + r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\s*%)?",
                       text, flags=re.IGNORECASE)
    conf = None
    if m_conf:
        c = float(m_conf.group(1))
        if m_conf.group(2):
            c /= 100.0
        conf = max(0.0, min(1.0, c))

    return rationale, prob, conf, False


def _detect_mode(df):
    """Return (mode, ensemble_response_cols_sorted)."""
    cols = set(df.columns)
    ensemble_cols = sorted(
        [c for c in cols if c.startswith("response_") and c[len("response_"):].isdigit()],
        key=lambda c: int(c[len("response_"):]),
    )
    if ensemble_cols and "response" in cols:
        raise ValueError(
            "Input has BOTH a `response` column and `response_*` columns; "
            "expected exactly one shape."
        )
    if ensemble_cols:
        return "ensemble", ensemble_cols
    if "response" in cols:
        return "single", []
    raise ValueError("Input lacks both `response` and `response_*` columns; "
                     f"got columns: {sorted(cols)}")


def _parse_column(series):
    """Parse a Series of raw response strings → (rationales, probs, confs, n_errors)."""
    rationales, probs, confs, errors = [], [], [], 0
    for resp in series:
        rat, p, c, err = parse_response(resp if isinstance(resp, str) else "")
        rationales.append(rat)
        probs.append(p)
        confs.append(c)
        if err:
            errors += 1
    return rationales, probs, confs, errors


def _report_brier(df, prob_col="prob"):
    if "resolution" in df.columns and prob_col in df.columns:
        y = np.array([1 if str(r).lower() == "yes" else 0 for r in df["resolution"]])
        p = np.array(df[prob_col]).astype(float)
        print(f"Brier on parsed {prob_col}: {((p - y) ** 2).mean():.4f}")


def process_single(df):
    """Default / w_conf: 1 response per row. Auto-promotes to w_conf if any
    parsed confidence is non-None."""
    rationales, probs, confs, errors = _parse_column(df["response"])
    df["rationale"] = rationales
    df["prob"] = probs
    has_conf = any(c is not None for c in confs)
    if has_conf:
        df["confidence"] = confs

    df["response"] = [
        {"decision_history": [{
            "reasoning": rat,
            "parameters": {"Yes": 100 * p, "No": 100 * (1 - p)},
        }]}
        for rat, p in zip(rationales, probs)
    ]

    mode_str = "w_conf" if has_conf else "default"
    print(f"[{mode_str}] Parsed {len(df)} entries; {errors} parse failures (default prob=0.5).")
    _report_brier(df)
    return df


def process_ensemble(df, response_cols):
    """Ensemble: parse each `response_k` → `prob_k`, `rationale_k`.
    Aggregate `prob = mean_k prob_k` (the ensemble baseline metric)."""
    n = len(response_cols)
    per_sample_probs = []
    per_sample_rationales = []
    total_errors = 0

    for col in response_cols:
        k = int(col[len("response_"):])
        rats, probs, _, errs = _parse_column(df[col])
        df[f"rationale_{k}"] = rats
        df[f"prob_{k}"] = probs
        per_sample_probs.append(probs)
        per_sample_rationales.append(rats)
        total_errors += errs

    # aggregate: mean prob across samples (NaN-safe over the n per-sample lists)
    probs_matrix = np.array(per_sample_probs, dtype=float)  # [n_samples, n_rows]
    df["prob"] = probs_matrix.mean(axis=0)
    # rationale: take the first sample's rationale as the representative
    df["rationale"] = per_sample_rationales[0]

    print(f"[ensemble] Parsed {len(df)} entries × {n} samples; "
          f"{total_errors} parse failures total (default prob=0.5).")
    _report_brier(df)
    return df


def process(input_path, output_path):
    df = pd.read_json(input_path, convert_dates=False)
    mode, ensemble_cols = _detect_mode(df)

    if mode == "ensemble":
        df = process_ensemble(df, ensemble_cols)
    else:
        df = process_single(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Raw verbalized output JSON (either `response` or `response_1..N`)")
    ap.add_argument("--output", required=True,
                    help="Where to write the processed JSON")
    args = ap.parse_args()
    process(args.input, args.output)
