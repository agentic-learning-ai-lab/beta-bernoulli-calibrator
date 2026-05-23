#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import get_metrics

"""
Global post-hoc calibration baselines: temperature scaling, Platt scaling, beta calibration.
Reads `prob` (parsed initial verbalized forecast) and writes calibrated `pred`.

Example (run from the project root):
INPUT_MODEL=qwen3-8b
python ./baselines/platt.py \
  --train_data ./results/default/verbalized/$INPUT_MODEL/train.json \
  --val_data   ./results/default/verbalized/$INPUT_MODEL/val.json \
  --target_data ./results/default/verbalized/$INPUT_MODEL/test.json \
  --output_dir ./results/default/platt/$INPUT_MODEL \
  --method platt   # one of: temp, platt, beta
"""


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# -----------------------
# Calibrators
# -----------------------

class BaseCalibrator:
    name: str = "base"
    def fit(self, p: np.ndarray, y: np.ndarray, device: torch.device, max_iter: int, lr: float):
        raise NotImplementedError
    def transform(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def meta(self) -> dict:
        return {}

class TemperatureScaling(BaseCalibrator):
    name = "temp"

    def __init__(self, init_T: float = 1.0):
        self.T = float(init_T)

    def fit(self, p: np.ndarray, y: np.ndarray, device: torch.device, max_iter: int, lr: float):
        p_t = torch.as_tensor(p, dtype=torch.float32, device=device).clamp(1e-6, 1 - 1e-6)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        z = torch.log(p_t) - torch.log1p(-p_t)

        logT = torch.nn.Parameter(torch.log(torch.tensor([self.T], dtype=torch.float32, device=device)))
        opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            T = torch.exp(logT)
            loss = F.binary_cross_entropy_with_logits(z / T, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(torch.exp(logT).item())
        return self

    @torch.no_grad()
    def transform(self, p: np.ndarray) -> np.ndarray:
        p_t = torch.as_tensor(p, dtype=torch.float32).clamp(1e-12, 1 - 1e-12)
        z = torch.log(p_t) - torch.log1p(-p_t)
        return torch.sigmoid(z / self.T).cpu().numpy()

    def meta(self) -> dict:
        return {"T": self.T}

class PlattScaling(BaseCalibrator):
    name = "platt"

    def __init__(self, init_a: float = 1.0, init_b: float = 0.0):
        self.a = float(init_a)
        self.b = float(init_b)

    def fit(self, p: np.ndarray, y: np.ndarray, device: torch.device, max_iter: int, lr: float):
        p_t = torch.as_tensor(p, dtype=torch.float32, device=device).clamp(1e-6, 1 - 1e-6)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        z = torch.log(p_t) - torch.log1p(-p_t)

        a = torch.nn.Parameter(torch.tensor([self.a], dtype=torch.float32, device=device))
        b = torch.nn.Parameter(torch.tensor([self.b], dtype=torch.float32, device=device))
        opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = a * z + b
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.a = float(a.item())
        self.b = float(b.item())
        return self

    @torch.no_grad()
    def transform(self, p: np.ndarray) -> np.ndarray:
        p_t = torch.as_tensor(p, dtype=torch.float32).clamp(1e-12, 1 - 1e-12)
        z = torch.log(p_t) - torch.log1p(-p_t)
        return torch.sigmoid(z * self.a + self.b).cpu().numpy()

    def meta(self) -> dict:
        return {"a": self.a, "b": self.b}

class BetaCalibration(BaseCalibrator):
    name = "beta"

    def __init__(self, init_A: float = 1.0, init_B: float = 1.0, init_C: float = 0.0, eps: float = 1e-6):
        self.A = float(init_A)
        self.B = float(init_B)
        self.C = float(init_C)
        self.eps = float(eps)

    def fit(self, p: np.ndarray, y: np.ndarray, device: torch.device, max_iter: int, lr: float):
        # p in (0,1)
        p_t = torch.as_tensor(p, dtype=torch.float32, device=device).clamp(self.eps, 1 - self.eps)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

        x1 = torch.log(p_t)          # log f(x)
        x2 = torch.log1p(-p_t)       # log(1 - f(x))

        A = torch.nn.Parameter(torch.tensor([self.A], dtype=torch.float32, device=device))
        B = torch.nn.Parameter(torch.tensor([self.B], dtype=torch.float32, device=device))
        C = torch.nn.Parameter(torch.tensor([self.C], dtype=torch.float32, device=device))

        opt = torch.optim.LBFGS([A, B, C], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = A * x1 + B * x2 + C
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            loss.backward()
            return loss

        opt.step(closure)

        self.A = float(A.item())
        self.B = float(B.item())
        self.C = float(C.item())
        return self

    @torch.no_grad()
    def transform(self, p: np.ndarray) -> np.ndarray:
        p_t = torch.as_tensor(p, dtype=torch.float32).clamp(self.eps, 1 - self.eps)
        x1 = torch.log(p_t)
        x2 = torch.log1p(-p_t)
        logits = self.A * x1 + self.B * x2 + self.C
        return torch.sigmoid(logits).cpu().numpy()

    def meta(self) -> dict:
        return {"A": self.A, "B": self.B, "C": self.C}

def make_calibrator(method: str, init_T: float) -> BaseCalibrator:
    if method == "temp":
        return TemperatureScaling(init_T=init_T)
    if method == "platt":
        return PlattScaling(init_a=1.0, init_b=0.0)
    if method == "beta":
        return BetaCalibration(init_A=1.0, init_B=1.0, init_C=0.0)
    raise ValueError(f"Unknown method: {method}")



# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Fit calibrator on val set and apply to target set.")
    ap.add_argument("--train_data", required=False, default=None)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--out_col", default="pred")
    ap.add_argument("--method", choices=["temp", "platt", "beta"], default="temp")
    ap.add_argument("--init_T", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = choose_device(args.device)

    config = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "target_data": args.target_data,
        "output_dir": args.output_dir,
        "out_col": args.out_col,
        "method": args.method,
        "init_T": args.init_T,
        "max_iter": args.max_iter,
        "lr": args.lr,
        "device": device.type,
    }
    print("Config:", config)

    # --- Fit on validation ---
    if args.train_data is not None:
        train_df = pd.read_json(args.train_data, convert_dates=False)
        val_df = pd.read_json(args.val_data, convert_dates=False)
        val_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        val_df = pd.read_json(args.val_data, convert_dates=False)
    y_val = [1 if i == 'yes' else 0 for i in val_df['resolution']]
    p_val = val_df["prob"].astype(float).to_numpy()

    p_fit = np.clip(p_val, 1e-6, 1 - 1e-6)
    y_fit = np.array(y_val).astype(float)

    cal = make_calibrator(args.method, args.init_T)
    cal.fit(p_fit, y_fit, device=device, max_iter=args.max_iter, lr=args.lr)

    print("Trained Params:", cal.meta())

    # --- Apply to target ---
    df = pd.read_json(args.target_data, convert_dates=False)
    p_target = df["prob"].astype(float).to_numpy()

    df[args.out_col] = cal.transform(p_target)
    if args.method == 'temp':
        df['T'] = cal.T
    elif args.method == 'platt':
        df['a'] = cal.a
        df['b'] = cal.b
    elif args.method == 'beta':
        df['a'] = cal.A
        df['b'] = cal.B
        df['c'] = cal.C

    # store params in columns for convenience
    for k, v in cal.meta().items():
        df[k] = v

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.target_data).split(".")[0]
    save_path = os.path.join(args.output_dir, f"{file_name}.json")
    df.to_json(save_path, orient="records")
    print(f"Wrote {args.out_col} to {save_path}")

    print("---- Before ----")
    get_metrics(df, "prob")

    print("---- After ----")
    get_metrics(df, args.out_col)


if __name__ == "__main__":
    main()
