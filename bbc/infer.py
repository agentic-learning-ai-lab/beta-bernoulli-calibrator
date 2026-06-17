#!/usr/bin/env python3
"""
Run BBC inference on a verbalized JSON and write per-question (alpha, beta, weight, pred).

Example (from the project root):
python ./bbc/infer.py \
    --model_dir  ./results/default/bbc/out_llama3_2-1b/in_claude-4-sonnet/binary-1-human-1 \
    --test_path  ./results/default/verbalized/claude-4-sonnet/test.json \
    --output_path ./results/default/bbc/out_llama3_2-1b/in_claude-4-sonnet/binary-1-human-1/test_with_preds.json \
    --batch_size 16
"""

import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils.metric import get_ece


def _format_prompt(entry, use_reasoning: bool, use_forecast: bool) -> str:
    r = entry["response"]["decision_history"][-1]
    q = entry["question"]

    if use_reasoning:
        step_reasoning = r.get("step_reasoning", "")
        reasoning = r["reasoning"]
        reasoning_text = f"Reasoning: {reasoning}\n\n----\n"
        step_text = ("Step reasoning:\n" + "\n".join(step_reasoning) + "\n\n----\n") if step_reasoning else ""
    else:
        reasoning_text, step_text = "", ""

    forecast_text = f"Forecast: {r['parameters']}\n" if use_forecast else ""

    text = f"Forecast Question: {q}\n\n----\n{reasoning_text}{step_text}{forecast_text}"
    text += "<endoftext>"
    return text

class LLMRegressionNetwork(nn.Module):
    """
      - Loads backbone via AutoModel.from_pretrained(model_dir)
      - Loads head from head.safetensors (lora/last) or checkpoint.safetensors (full)
    """
    def __init__(self, model_name: str, finetune_method: str, hidden_size: int = 256, n_models: int = 1):
        super().__init__()
        self.backbone = None
        self._model_name = model_name
        self.finetune_method = finetune_method
        self.hidden_size = hidden_size
        self.head = None
        self.input_device = None
        self.n_models = n_models

    def build_head(self, llm_hidden_size: int, device: torch.device):
        self.head = nn.Sequential(
            nn.Linear(llm_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_models * 3)
        ).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-2]
        last_index = (attention_mask.sum(dim=1) - 1).clamp(min=0).to(torch.long)
        sel = torch.arange(hidden.size(0), device=hidden.device)
        cls_token_output = hidden[sel, last_index, :]
        z = self.head(cls_token_output.to(self.head[0].weight.device))
        z = z.view(-1, self.n_models, 3)                                # [B, K, 3]
        alpha = F.softplus(z[:, :, 0]) + 1                              # [B, K]
        beta  = F.softplus(z[:, :, 1]) + 1                              # [B, K]
        w     = torch.softmax(z[:, :, 2], dim=-1)                       # [B, K]
        return alpha, beta, w

    @torch.no_grad()
    def get_prob(self, input_ids, attention_mask, return_ab=False):
        alpha, beta, w = self.forward(input_ids, attention_mask)     # [B,K] each
        means = alpha / (alpha + beta + 1e-12)                       # [B,K]
        p = (means * w).sum(dim=1)                                   # [B]
        return (p, alpha, beta, w) if return_ab else p

    def load_from_dir(self, model_dir: str):
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True, device_map="auto")
        some_param = next(self.backbone.parameters())
        self.input_device = some_param.device
        self.build_head(self.backbone.config.hidden_size, self.input_device)

        head_path = os.path.join(model_dir, "head.safetensors")
        full_ckpt = os.path.join(model_dir, "checkpoint.safetensors")
        if os.path.exists(head_path):
            state = load_file(head_path)
            self.head.load_state_dict(state)
            print(f"[infer] Loaded head: {head_path}")
        elif os.path.exists(full_ckpt):
            state = load_file(full_ckpt)
            self.load_state_dict(state)
            print(f"[infer] Loaded full checkpoint: {full_ckpt}")
        else:
            raise FileNotFoundError(
                f"Missing head.safetensors/checkpoint.safetensors in {model_dir}"
            )
        self.eval()

def load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)

def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def compute_metrics(y_true: List[int], y_prob: List[float]):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).clip(0.0, 1.0)
    brier = np.mean((y_prob - y_true) ** 2)
    acc = np.mean((y_prob >= 0.5) == (y_true == 1))
    auc = roc_auc_score(y_true, y_prob)
    ece = get_ece(y_true, y_prob, n_bins=10)
    return {
        "brier": float(brier),
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "ece_10bins": float(ece),
        "n": int(y_true.shape[0]),
        "pos_rate": float(y_true.mean() if y_true.size else 0.0),
    }

def main():
    parser = argparse.ArgumentParser(description="Run inference on final test set (with metrics)")
    parser.add_argument("--model_dir", type=str, required=True, help="Trained model output folder (contains config.json & weights)")
    parser.add_argument("--test_path", type=str, required=True, help="Test JSON path")
    parser.add_argument("--output_path", type=str, required=True, help="Where to write test JSON with predictions")
    parser.add_argument("--metrics_path", type=str, default="", help="Optional metrics JSON path (default = <output_dir>/metrics.json)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--truncate", action="store_true", help="Enable tokenizer truncation (default mirrors training: False)")
    args = parser.parse_args()

    cfg = load_config(args.model_dir)
    model_name = cfg["model_name"]
    use_reasoning = cfg.get("use_reasoning", True)
    use_forecast  = cfg.get("use_forecast", True)
    n_models = int(cfg.get("n_models", 1))

    print("[infer] Loaded config:", {
        "model_name": model_name,
        "use_reasoning": use_reasoning,
        "use_forecast": use_forecast,
        "n_models": n_models,
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    net = LLMRegressionNetwork(
        model_name=model_name,
        finetune_method=cfg.get("finetune_method", "lora"),
        hidden_size=cfg.get("hidden_size", 256),
        n_models=n_models,
    )
    net.load_from_dir(args.model_dir)

    # Load test set and build prompts
    test_entries = read_json(args.test_path)
    texts: List[str] = []
    y_true: List[int] = []
    has_resolution = all("resolution" in e for e in test_entries)
    for e in test_entries:
        texts.append(_format_prompt(e, use_reasoning, use_forecast))
        if has_resolution:
            y_true.append(1 if str(e["resolution"]).strip().lower() in ["yes", "yes.", "y", "true", "1"] else 0)

    # Batched inference
    preds: List[float] = []
    alphas_all = []
    betas_all  = []
    weights_all = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="Infer"):
            batch_texts = texts[start:start + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=bool(args.truncate),
            )
            p, a, b, w = net.get_prob(inputs["input_ids"], inputs["attention_mask"], return_ab=True)
            preds.extend(p.cpu().tolist())
            alphas_all.extend(a.detach().cpu().tolist())   # each is list length K
            betas_all.extend(b.detach().cpu().tolist())
            weights_all.extend(w.detach().cpu().tolist())

    # Attach predictions
    for e, p, a, b, w in zip(test_entries, preds, alphas_all, betas_all, weights_all):
        e["pred"]  = float(p)
        e["alpha"] = a
        e["beta"]  = b
        e["weight"] = w

    # Write predictions JSON
    write_json(args.output_path, test_entries)
    print(f"[infer] Wrote predictions to {args.output_path}")

    # Compute & write metrics only when labels are available.
    if has_resolution:
        metrics = compute_metrics(y_true, preds)
        metrics_path = args.metrics_path or os.path.join(os.path.dirname(args.output_path), "metrics.json")
        write_json(metrics_path, metrics)
        print("[infer] Test metrics:")
        for k, v in metrics.items():
            print(f"  {k:>10}: {v}")
        print(f"[infer] Wrote metrics to {metrics_path}")
    else:
        print("[infer] No `resolution` field found; skipped metrics.")

if __name__ == "__main__":
    main()
