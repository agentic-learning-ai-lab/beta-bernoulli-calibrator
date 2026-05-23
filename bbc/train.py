"""
Train the Beta-Bernoulli Calibrator.

Example (run from the project root):

python ./bbc/train.py \
  --train_path ./results/default/verbalized/claude-4-sonnet/train.json \
  --val_path   ./results/default/verbalized/claude-4-sonnet/val.json  \
  --test_path  ./results/default/verbalized/claude-4-sonnet/test.json \
  --net meta-llama/Llama-3.2-1B \
  --epochs 15 \
  --binary_coeff 1 --human_coeff 1 \
  --lr 1e-6 --lora_rank 256 --n_models 5 --print_interval 300 \
  --output ./results/default/bbc/out_llama3_2-1b/in_claude-4-sonnet/binary-1-human-1
"""

import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import save_file, load_file
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils.meter import AverageMeter
from utils.metric import get_ece

hf_home_path = os.getenv('HF_HOME')
if hf_home_path is not None:
  print(f"HF_HOME: {hf_home_path}")
else:
  print("HF_HOME environment variable is not set.")


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)   # Python hash seed
    random.seed(seed)                          # Python RNG
    np.random.seed(seed)                       # NumPy RNG
    torch.manual_seed(seed)                    # Torch CPU RNG
    torch.cuda.manual_seed_all(seed)           # All CUDA devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Beta negative log-pdf; used for the human-forecast term.
def beta_nll_loss(alphas, betas, targets):
    dist = D.Beta(alphas, betas)
    return -dist.log_prob(targets)


def save(config, trained_net=None):
  output_folder = config["output_folder"]
  if trained_net:
    # save model weights
    if config["finetune_method"] == "lora":
      trained_net.backbone.save_pretrained(output_folder)
      fname = os.path.join(output_folder, "head.safetensors")
      save_file(trained_net.head.state_dict(), fname)
    elif config["finetune_method"] == "last":
      fname = os.path.join(output_folder, "head.safetensors")
      save_file(trained_net.head.state_dict(), fname)
    else:
      fname = os.path.join(output_folder, "checkpoint.safetensors")
      save_file(trained_net.state_dict(), fname)
    print(f"Checkpoint saved to {fname}")


class LLMRegressionNetwork(nn.Module):

  def __init__(self, config):
    super(LLMRegressionNetwork, self).__init__()
    self.config = config
    self.backbone = AutoModel.from_pretrained(config["model_name"],
                                              trust_remote_code=True,
                                              device_map="auto")
    self.n_models = config["n_models"]

    # Freeze the backbone's parameters so we only train our new layers
    if config["finetune_method"] in ["last"]:
      print("--- Applying frozen backbone  ---")
      for param in self.backbone.parameters():
        param.requires_grad = False
    elif config["finetune_method"] in ["lora"]:
      print("--- Applying LoRA configuration ---")
      # Define the LoRA configuration
      peft_config = LoraConfig(
          r=config["lora_rank"],
          lora_alpha=config["lora_alpha"],
          target_modules=config["lora_target_modules"],
          lora_dropout=config["lora_dropout"],
          bias="none",
          task_type=TaskType.FEATURE_EXTRACTION)
      # Wrap the base model with the PEFT config
      self.backbone = get_peft_model(self.backbone, peft_config)
      print("Trainable parameters with LoRA:")
      self.backbone.print_trainable_parameters()

    # Define our new "head" that will be trained
    # It takes the LLM's output and maps it to our alpha and beta parameters
    llm_output_size = self.backbone.config.hidden_size
    param_list = list(self.backbone.parameters())
    head_device = param_list[-1].device
    self.head = nn.Sequential(
        nn.Linear(llm_output_size, config["hidden_size"]),
        nn.ReLU(),
        nn.Linear(config["hidden_size"], self.n_models * 3) # (α, β, weight) × K
    ).to(head_device)
    self.input_device = param_list[0].device

  def forward(self, input_ids, attention_mask):
    # Get the hidden states from the LLM backbone
    _input_ids = input_ids.to(self.input_device)
    _attention_mask = attention_mask.to(self.input_device)
    outputs = self.backbone(input_ids=_input_ids,
                            attention_mask=_attention_mask,
                            output_hidden_states=True,
                            use_cache=False)

    hidden = outputs.hidden_states[-2] # [B, T, H] (batch size, sequence length, hidden size), second to last layer
    last_index = (_attention_mask.sum(dim=1) - 1).clamp(min=0).to(torch.long) # [B]
    sel = torch.arange(hidden.size(0), device=hidden.device) # [0, 1, 2, ..., B-1]
    cls_token_output = hidden[sel, last_index, :]  # [B, H]
    z = self.head(cls_token_output.to(self.head[0].weight.device))

    z = z.view(-1, self.n_models, 3)     # [B, K, 3]
    alpha_raw  = z[:, :, 0]
    beta_raw   = z[:, :, 1]
    weight_raw = z[:, :, 2]

    # Shift by 1 so each Beta component has alpha, beta >= 1 (unimodal).
    alpha = 1.0 + F.softplus(alpha_raw)
    beta  = 1.0 + F.softplus(beta_raw)

    weights = torch.softmax(weight_raw, dim=-1)

    return alpha, beta, weights

  def get_prob(self, input_ids, attention_mask, return_ab=False):
    alpha, beta, w = self.forward(input_ids, attention_mask)
    means = alpha / (alpha + beta)
    mixture_mean = (means * w).sum(dim=1)
    return (mixture_mean, alpha, beta, w) if return_ab else mixture_mean

  def load(self, folder):
    backbone = AutoModel.from_pretrained(folder, device_map='auto')
    self.backbone = backbone
    head = load_file(os.path.join(folder, 'head.safetensors'))
    self.head.load_state_dict(head)


class ForecastTraceEnvironment:
  def __init__(self, config, tokenizer, data_path):
    self.tokenizer = tokenizer

    # Per-question training examples: (prompt text, outcome, [human histogram]).
    self.text_data = []
    results = json.load(open(data_path, "r"))
    self.config = config

    yes_count = 0
    total_count = 0

    for entry in results:
      a = float(entry["resolution"]=="yes")
      if a == 1.0:
        yes_count += 1
      total_count += 1
      text = self.format_prompt(entry)
      # Print the first formatted prompt as a sanity check.
      if len(self.text_data) == 0:
        print(text)
      if config["human_coeff"] > 0:
        human_label = np.array(entry["forecast_histogram"])
        if self.config["smooth_hist"]:
          human_label = gaussian_filter1d(human_label, sigma=1)
        if human_label.sum() == 0:
          # Fall back to a uniform histogram when no human forecasts exist.
          human_label = np.ones_like(human_label, dtype=np.float32) / len(human_label)
          print("Zero-sum at load time, entry id:", entry.get("question", "N/A"))
        self.text_data.append((text, a, human_label))
      else:
        self.text_data.append((text, a))
    self.num_items = len(self.text_data)
    print("--- Text Environment Initialized ---")
    print("yes ratio: ", yes_count / total_count)

  def format_prompt(self, entry):
    r = entry["response"]["decision_history"][-1]
    q = entry["question"]
    reasoning = r["reasoning"]
    parameters = r["parameters"]

    if self.config["use_reasoning"]:
      reasoning_text = f"Reasoning: {reasoning}\n" + f"\n----\n"
    else:
      reasoning_text = ""

    if self.config["use_forecast"]:
      forecast_text = f"Forecast: {parameters}\n"
    else:
      forecast_text = ""

    text = (f"Forecast Question: {q}\n" + f"\n----\n" + reasoning_text + forecast_text)
    text += "<endoftext>"

    return text

  def get_batch(self, batch_size):
    # Sample random indices from our dataset
    indices = np.random.randint(0, self.num_items, size=batch_size)

    # Get the text and true probabilities for the sampled indices
    batch_texts = [self.text_data[i][0] for i in indices]

    # Tokenize the text
    inputs = self.tokenizer(batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512)

    labels = torch.tensor([self.text_data[i][1] for i in indices]).unsqueeze(1)

    if len(self.text_data[0]) == 3:
      human_labels = torch.tensor(np.array([self.text_data[i][2] for i in indices]), dtype=torch.float32)
      return inputs, labels, human_labels
    else:
      return inputs, labels


def train(config):
  # Load the tokenizer for our chosen model
  tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  train_eval_env = ForecastTraceEnvironment(config, tokenizer, config["train_dataset_path"])
  val_eval_env = ForecastTraceEnvironment(config, tokenizer, config["val_dataset_path"])
  test_eval_env = None

  if config.get("test_dataset_path"):
    test_eval_env = ForecastTraceEnvironment(config, tokenizer, config["test_dataset_path"])

  net = LLMRegressionNetwork(config)

  optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
  
  output_folder = config["output_folder"]

  # Create a fixed evaluation set to monitor true progress
  # Initialize average meters
  loss_meter = AverageMeter()

  train_brier_meter = AverageMeter()
  train_acc_meter   = AverageMeter()
  train_auc_meter   = AverageMeter()
  train_ece_meter = AverageMeter()

  val_brier_meter = AverageMeter()
  val_acc_meter   = AverageMeter()
  val_auc_meter   = AverageMeter()
  val_ece_meter = AverageMeter()

  if test_eval_env is not None:
    test_brier_meter = AverageMeter()
    test_acc_meter = AverageMeter()
    test_auc_meter = AverageMeter()
    test_ece_meter = AverageMeter()

  batch_history, epoch_history = [], []
  loss_history = []
  train_brier_history, train_acc_history, train_auc_history, train_ece_history = [], [], [], []
  val_brier_history, val_acc_history, val_auc_history, val_ece_history = [], [], [], []
  if test_eval_env is not None:
    test_brier_history, test_acc_history, test_auc_history, test_ece_history = [], [], [], []


  print("\n--- Starting Training ---")
  num_batches = len(train_eval_env.text_data) * config["epochs"] // config["batch_size"]
  best_val_brier = float("inf")
  best_epoch = -1

  # save config
  json.dump(config, open(f"{output_folder}/config.json", "w"))

  for i in tqdm(range(num_batches), desc="Training"):
    net.train()
    batch = train_eval_env.get_batch(config["batch_size"])
    inputs, true_probs = batch[0], batch[1]

    alphas, betas, w = net(inputs['input_ids'], inputs['attention_mask'])  # [B,K], [B,K], [B,K]
    y = true_probs[:, 0].to(device=alphas.device, dtype=alphas.dtype)

    loss = 0.0

    if config["binary_coeff"] > 0:
      probs_yes = alphas / (alphas + betas + 1e-8)      # [B,K]
      mixture_p = (w * probs_yes).sum(dim=1)            # [B]
      mixture_p = torch.clamp(mixture_p, 1e-6, 1 - 1e-6)

      y_bin = y.view(-1)                                # [B]
      nll = F.binary_cross_entropy(mixture_p, y_bin, reduction="mean")
      loss = loss + config["binary_coeff"] * nll

    if config["human_coeff"] > 0:
      human_histo = batch[2].to(device=alphas.device, dtype=alphas.dtype)
      dist = human_histo / human_histo.sum(dim=-1, keepdim=True)
      dist_val = torch.linspace(0.005, 0.995, steps=100,
                                device=alphas.device, dtype=alphas.dtype).view(1, -1)

      # pdfs: [B,K,bins]  (using exp(-nll) because beta_nll_loss returns -log pdf)
      pdfs = torch.exp(-beta_nll_loss(alphas.unsqueeze(-1), betas.unsqueeze(-1), dist_val))
      mix_pdf = (pdfs * w.unsqueeze(-1)).sum(dim=1) + 1e-9  # [B,bins]

      human_loss = -(dist * torch.log(mix_pdf)).sum(dim=-1).mean()
      loss = loss + config["human_coeff"] * human_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update meters
    loss_meter.update(loss.item())

    def full_eval_next(env, brier_meter, acc_meter, auc_meter, ece_meter, batch_size):
      # reset meters for this pass
      brier_meter.reset(); acc_meter.reset(); auc_meter.reset(); ece_meter.reset()

      net.eval()
      all_p, all_y = [], []
      all_a, all_b = [], []
      all_w = []
      with torch.no_grad():
          for start in range(0, env.num_items, batch_size):
              end = min(start + batch_size, env.num_items)

              # gather this chunk
              batch_texts = [env.text_data[i][0] for i in range(start, end)]
              y = torch.tensor([env.text_data[i][1] for i in range(start, end)],
                                        dtype=torch.float32)

              # tokenize and predict
              inputs = tokenizer(batch_texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512)

              pred, alpha, beta, w = net.get_prob(inputs["input_ids"], inputs["attention_mask"], return_ab=True)
              pred = pred.cpu()
              pred = torch.clamp(pred, 0.0, 1.0)

              # per-chunk metrics, weighted by chunk size
              n = y.shape[0]
              p_np = pred.numpy()
              y_np = y.numpy().astype(int)
              brier = ((p_np - y_np)**2).mean()
              acc   = ((p_np >= 0.5) == y_np).mean()

              brier_meter.update(brier, n=n)
              acc_meter.update(acc, n=n)

              all_p.append(p_np); all_y.append(y_np)
              if alpha is not None and beta is not None and w is not None:
                all_a.append(alpha.cpu())
                all_b.append(beta.cpu())
                all_w.append(w.cpu())

      # AUC must be computed once on the full set
      all_p = np.concatenate(all_p, axis=0)
      all_y = np.concatenate(all_y, axis=0)
      all_a = torch.cat(all_a, dim=0) if len(all_a)>0 else None
      all_b = torch.cat(all_b, dim=0) if len(all_b)>0 else None
      all_w = torch.cat(all_w, dim=0) if len(all_w)>0 else None

      try:
          auc = roc_auc_score(all_y, all_p)
          auc_meter.update(float(auc), n=1)  # single value, no per-batch averaging

          ece = get_ece(all_y, all_p, n_bins=10)
          ece_meter.update(float(ece), n=1)

      except ValueError:
          # happens if only one class present; skip updating AUC
          pass
      return all_p, all_y, all_a, all_b, all_w

    if (i + 1) % config["print_interval"] == 0:
      net.eval()
      with torch.no_grad():
        train_p, train_y, train_a, train_b, train_w = full_eval_next(train_eval_env, train_brier_meter, train_acc_meter, train_auc_meter, train_ece_meter, config["eval_batch_size"])
        val_p, val_y, val_a, val_b, val_w = full_eval_next(val_eval_env, val_brier_meter, val_acc_meter, val_auc_meter, val_ece_meter, config["eval_batch_size"])  
        if test_eval_env is not None:
          test_p, test_y, test_a, test_b, test_w = full_eval_next(test_eval_env, test_brier_meter, test_acc_meter, test_auc_meter, test_ece_meter, config["eval_batch_size"])

      epoch = i // (len(train_eval_env.text_data) // config["batch_size"]) + 1
      now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

      print(
          f"[{now_str}] Batch {i+1}/{num_batches} | Epoch {epoch} | Loss={loss_meter.avg:.4f} \n"
          f" - train_brier={train_brier_meter.avg:.4f} train_acc={train_acc_meter.avg:.4f} train_auc={(train_auc_meter.avg if train_auc_meter.count>0 else float('nan')):.4f} train_ece={train_ece_meter.avg:.4f} \n"
          f" - val_brier={val_brier_meter.avg:.4f} val_acc={val_acc_meter.avg:.4f} val_auc={(val_auc_meter.avg if val_auc_meter.count>0 else float('nan')):.4f} val_ece={val_ece_meter.avg:.4f} \n"
          + (
              (
                  f" - test_brier={test_brier_meter.avg:.4f} test_acc={test_acc_meter.avg:.4f} "
                  f"test_auc={(test_auc_meter.avg if test_auc_meter.count>0 else float('nan')):.4f} "
                  f"test_ece={test_ece_meter.avg:.4f} \n"
              )
              if test_eval_env is not None else ""
          )
      )

      # Store history for plotting
      batch_history.append(i+1)
      epoch_history.append(epoch)
      loss_history.append(loss_meter.avg)

      train_brier_history.append(train_brier_meter.avg)
      train_acc_history.append(train_acc_meter.avg)
      train_auc_history.append(train_auc_meter.avg if train_auc_meter.count>0 else float('nan'))
      train_ece_history.append(train_ece_meter.avg)

      val_brier_history.append(val_brier_meter.avg)
      val_acc_history.append(val_acc_meter.avg)
      val_auc_history.append(val_auc_meter.avg if val_auc_meter.count>0 else float('nan'))
      val_ece_history.append(val_ece_meter.avg)

      if test_eval_env is not None:
        test_brier_history.append(test_brier_meter.avg)
        test_acc_history.append(test_acc_meter.avg)
        test_auc_history.append(test_auc_meter.avg if test_auc_meter.count>0 else float('nan'))
        test_ece_history.append(test_ece_meter.avg)

      if val_brier_meter.avg < best_val_brier:
        best_val_brier = val_brier_meter.avg
        best_epoch = epoch

        # save the best model and val dataset with preds
        results = json.load(open(config["val_dataset_path"], "r"))
        preds = val_p.tolist()
        if val_a is not None and val_b is not None and val_w is not None:
          for entry, p, a, b, w in zip(results, preds, val_a, val_b, val_w):
              entry["pred"] = float(p)
              entry["alpha"] = a.tolist()
              entry["beta"]  = b.tolist()
              entry["weight"] = w.tolist()
        else:
          for entry, p in zip(results, preds):
              entry["pred"] = float(p)

        save(config, trained_net=net)
        json.dump(results, open(f"{output_folder}/val_dataset_with_preds.json", "w"))
        print(f"=== New best model and val dataset at batch {i+1} epoch {epoch} (val_brier={best_val_brier:.4f}) saved ===")

        # also save test dataset with preds (if any)
        if test_eval_env is not None:
          test_results = json.load(open(config["test_dataset_path"], "r"))
          test_preds = test_p.tolist()
          if test_a is not None and test_b is not None and test_w is not None:
            for entry, p, a, b, w in zip(test_results, test_preds, test_a, test_b, test_w):
              entry["pred"]  = float(p)
              entry["alpha"] = a.tolist()
              entry["beta"]  = b.tolist()
              entry["weight"] = w.tolist()
          else:
            for entry, p in zip(test_results, test_preds):
              entry["pred"] = float(p)

          output_path = f"{output_folder}/test_dataset_with_preds.json"
          json.dump(test_results, open(output_path, "w"))
          print(f"=== test dataset (with preds) saved to {output_path} ===")

        # also save training dataset with preds
        train_results = json.load(open(config["train_dataset_path"], "r"))
        train_preds = train_p.tolist()
        if train_a is not None and train_b is not None and train_w is not None:
          for entry, p, a, b, w in zip(train_results, train_preds, train_a, train_b, train_w):
              entry["pred"]  = float(p)
              entry["alpha"] = a.tolist()
              entry["beta"]  = b.tolist()
              entry["weight"] = w.tolist()
        else:
          for entry, p in zip(train_results, train_preds):
              entry["pred"] = float(p)

        train_output_path = f"{output_folder}/train_dataset_with_preds.json"
        json.dump(train_results, open(train_output_path, "w"))
        print(f"=== Train dataset (with preds) saved to {train_output_path} ===")

      loss_meter.reset()

    history = {
      "batch": batch_history,
      "epoch": epoch_history,
      "loss": loss_history,

      "brier": train_brier_history,
      "acc": train_acc_history,
      "auc": train_auc_history,
      "ece": train_ece_history,

      "val_brier": val_brier_history,
      "val_acc": val_acc_history,
      "val_auc": val_auc_history,
      "val_ece": val_ece_history,
    }

    if test_eval_env is not None:
      history.update({
          "test_brier": test_brier_history,
          "test_acc": test_acc_history,
          "test_auc": test_auc_history,
          "test_ece": test_ece_history,
      })

    json.dump(history, open(f"{output_folder}/history.json", "w"))

  # Save last-epoch train predictions from the final model.
  print("=== Saving last-epoch predictions for train (final model) ===")

  train_results_last = json.load(open(config["train_dataset_path"], "r"))
  train_p_last, train_y_last, train_a_last, train_b_last, train_w_last = full_eval_next(
      train_eval_env,
      train_brier_meter, train_acc_meter, train_auc_meter, train_ece_meter,
      config["eval_batch_size"]
  )

  if train_a_last is not None and train_b_last is not None and train_w_last is not None:
    for entry, p, a, b, w in zip(train_results_last, train_p_last, train_a_last, train_b_last, train_w_last):
      entry["pred"]  = float(p)
      entry["alpha"] = a.tolist()
      entry["beta"]  = b.tolist()
      entry["weight"] = w.tolist()
  else:
    for entry, p in zip(train_results_last, train_p_last):
      entry["pred"] = float(p)

  train_output_last = f"{output_folder}/train_dataset_with_preds_last_epoch.json"
  json.dump(train_results_last, open(train_output_last, "w"))
  print(f"=== Final last-epoch train dataset saved to {train_output_last} ===")

  print("--- Training Finished ---")
  return net, optimizer, tokenizer, history


# --- Configuration ---
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train regression network")
  parser.add_argument("--no_reasoning", action="store_true", help="Remove reasoning from input prompt")
  parser.add_argument("--no_forecast", action="store_true", help="Remove forecast from input prompt")
  parser.add_argument("--net", type=str, default="Qwen/Qwen2.5-0.5B", help="Network name")
  parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
  parser.add_argument("--print_interval", type=int, default=100, help="Print interval")
  parser.add_argument("--train_path", type=str, help="Training dataset path")
  parser.add_argument("--val_path", type=str, help="Eval dataset path")
  parser.add_argument("--test_path", type=str, default="", help="Optional testence dataset path for end-of-epoch eval")
  parser.add_argument("--n_models", type=int, default=5, help="Number of mixture beta models")
  parser.add_argument("--binary_coeff", type=float, default=1.0, help="Binary loss coefficient")
  parser.add_argument("--human_coeff", type=float, default=0, help="Loss coefficient for human forecast")
  parser.add_argument("--finetune_method", type=str, default="lora", help="Fine-tuning method")
  parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
  parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank")
  parser.add_argument("--lora_dropout", type=float, default=0.2, help="LoRA dropout")
  parser.add_argument("--lora_target", type=str, default="attn", help="LoRA target")
  parser.add_argument("--output", type=str, help="Output folder")
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay coefficient")
  parser.add_argument("--not_smooth_hist", action="store_true", help="Not to smooth human histogram")
  parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

  args = parser.parse_args()
  set_all_seeds(args.seed)
  if args.lora_target == 'attn':
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
  elif args.lora_target == 'mlp':
    target_modules = ["gate_proj", "up_proj", "down_proj"]
  elif args.lora_target == 'all':
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

  config = {
      "model_name": args.net,
      "output_folder": args.output,
      "train_dataset_path": args.train_path,
      "val_dataset_path": args.val_path,
      "learning_rate": args.lr, 
      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "eval_batch_size": args.batch_size,
      "eval_num_batches": 50,
      "hidden_size": 256,  # Size of the layer between LLM and output
      "entropy_coefficient": 0.00,
      "print_interval": args.print_interval,
      "epsilon": 1e-5,
      "finetune_method": args.finetune_method,  # last, lora, full
      "lora_rank": args.lora_rank,
      "lora_alpha": args.lora_rank,
      "lora_dropout": args.lora_dropout,
      "lora_target_modules": target_modules,
      "use_baseline": False,
      "use_reasoning": not args.no_reasoning,
      "use_forecast": not args.no_forecast,
      "binary_coeff": args.binary_coeff,
      "human_coeff": args.human_coeff,
      "weight_decay": args.weight_decay,
      "test_dataset_path": args.test_path, 
      "smooth_hist": not args.not_smooth_hist,
      "seed": args.seed,
      "n_models": args.n_models,
  }
  print(config)

  output_folder = config["output_folder"]
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  trained_net, optimizer, tokenizer, history = train(config)
  json.dump(history, open(f"{output_folder}/history.json", "w"))