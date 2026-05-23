import argparse
import pandas as pd
import os
from tqdm import tqdm
import math
import re

from vllm import LLM
from vllm.sampling_params import SamplingParams


"""
P(True) baseline: prompt the model to answer Yes/No, derive p_yes from next-token logits.
Whitebox-only (requires logit access via vLLM).

Example (run from the project root):
# Direct first-token probability (no rationale)
python baselines/ptrue.py \
  --model_name llama3-8b \
  --mode direct \
  --input_path ./data/default/test.json \
  --output_path ./results/default/ptrue

# Two-pass rationale mode
python baselines/ptrue.py \
  --model_name llama3-8b \
  --mode rationale \
  --input_path ./data/default/test.json \
  --output_path ./results/default/ptrue

Set HF_HOME to control where vLLM caches model weights.
"""


# =========================
# Constants
# =========================

MODEL_MAPPING = {
    'llama3-8b':   'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3-70b':  'meta-llama/Llama-3.3-70B-Instruct',
    'gemma-3-12b': 'google/gemma-3-12b-it',
    'gemma-3-27b': 'google/gemma-3-27b-it',
    'qwen2_5-7b':  'Qwen/Qwen2.5-7B-Instruct',
    'qwen2_5-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen3-8b':    'Qwen/Qwen3-8B',
    'qwen3-32b':   'Qwen/Qwen3-32B',
}

LARGE_MODELS = {'llama3-70b', 'qwen2_5-72b'}

SYSTEM_MSG = ""

# Keep the same variants; the next-token scorer will automatically ignore multi-token variants.
YES_VARS = [" Yes", "Yes", "\nYes"]
NO_VARS  = [" No",  "No",  "\nNo"]


# =========================
# Model loading
# =========================

def load_model(config):
    model_name = config['model_name']
    model_id = config['model_id']

    kwargs = dict(
        model=model_id,
        tensor_parallel_size=config['num_gpu'],
        max_model_len=4096,
    )
    if model_name in LARGE_MODELS:
        kwargs.update(dtype='float16')
    elif 'gemma' in model_name:
        kwargs.update(dtype='bfloat16')
        kwargs.pop('max_model_len', None)
    else:
        kwargs.update(dtype='float16')
    return LLM(**kwargs)


# =========================
# Prompt helpers
# =========================

def build_chat_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        return user_prompt + "\n\n"


def get_prompt(row):
    return f"""You're an expert in forecasting events. Make a prediction of whether the question will be resolved as true. You MUST give a "Yes"/"No" answer UNDER ALL CIRCUMSTANCES.

To support your reasoning, recall relevant recent events, facts, or widely known information.

Once you have completed your reasoning, output your answer as a single word "Yes" or "No".

Question: {row['question']}
Resolution Criteria: {row['resolution_criteria']}

Today’s date: {row['open_date']}
Question close date: {row['close_date']}

Please follow the output format:
[Rationale:]
[Answer:]"""


def get_prompt_direct(row):
    return f"""You're an expert in forecasting events. Make a prediction of whether the question will be resolved as true.
You MUST answer with a single word: Yes or No.

Question: {row['question']}
Resolution Criteria: {row['resolution_criteria']}

Today’s date: {row['open_date']}
Question close date: {row['close_date']}

Answer:"""


# =========================
# Generation (rationale mode)
# =========================

def generate_response(client, prompt_ls, max_tokens=1024):
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
    outputs = client.generate(prompt_ls, sampling_params=sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


# =========================
# Log-prob utilities
# =========================

def _logsumexp(vals):
    m = max(vals)
    if m == -float('inf'):
        return -float('inf')
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _extract_token_logprob(entry, token_id):
    """
    Supports vLLM formats:
      - dict[token_id] -> LogprobObj (has .logprob) or float
      - list[LogprobObj] where each has .token_id and .logprob
    """
    if entry is None:
        return -float('inf')
    if isinstance(entry, dict) and token_id in entry:
        v = entry[token_id]
        return float(v.logprob) if hasattr(v, "logprob") else float(v)
    if isinstance(entry, list):
        for v in entry:
            if getattr(v, "token_id", None) == token_id:
                return float(v.logprob)
    return -float('inf')


def _get_step0_logprobs_from_output(out):
    """
    Returns the next-token logprob structure for step 0:
      - dict or list (depending on vLLM version)
    """
    if not out.outputs:
        return None
    if not hasattr(out.outputs[0], "logprobs"):
        return None

    lp = out.outputs[0].logprobs

    # vLLM may return:
    # - list (per generated step) of dicts/lists
    # - a dict/list directly for the step
    if isinstance(lp, list) and len(lp) > 0 and isinstance(lp[0], (dict, list)):
        return lp[0]
    return lp


# =========================
# Next-token probability modes (NEW)
# =========================

def class_logprob_nexttoken_vllm(client, tokenizer, prefix_text, variants, topk=20):
    """
    Computes log P(class | prefix) by summing probabilities of variants
    from the NEXT-TOKEN distribution.

    Only variants that tokenize to exactly 1 token can be used here.
    Multi-token variants are ignored.
    """
    # Get next-token distribution after prefix
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=topk)
    out = client.generate([prefix_text], sampling_params=sp, use_tqdm=False)[0]
    step0 = _get_step0_logprobs_from_output(out)

    lps = []
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) != 1:
            continue  # can't score multi-token variants from a 1-step distribution
        tid = ids[0]
        lps.append(_extract_token_logprob(step0, tid))

    if not lps:
        return -float('inf')
    return _logsumexp(lps)


def get_yes_probability_direct_vllm(client, tokenizer, user_prompt, topk=20):
    """
    Direct mode: compute p_yes from NEXT-TOKEN distribution after the chat prefix.
    Minimal change: keep function name/signature similar, just swap internals.
    """
    prefix = build_chat_prompt(tokenizer, user_prompt)

    lp_yes = class_logprob_nexttoken_vllm(client, tokenizer, prefix, YES_VARS, topk=topk)
    lp_no  = class_logprob_nexttoken_vllm(client, tokenizer, prefix, NO_VARS,  topk=topk)

    if lp_yes == -float('inf') and lp_no == -float('inf'):
        return 0.5, lp_yes, lp_no

    # numerically stable normalization
    m = max(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    return p_yes, lp_yes, lp_no


def parse_rationale(gen_text):
    m = list(re.finditer(r"\[\s*Answer\s*:\s*\]", gen_text, flags=re.IGNORECASE))
    if m:
        return gen_text[:m[-1].start()].strip()
    return gen_text.strip()


def get_yes_probability_w_rationale_vllm(client, tokenizer, user_prompt, gen_text, topk=20):
    """
    Rationale mode: keep existing two-pass behavior, but replace scoring step
    with NEXT-TOKEN scoring at the position right after '[Answer:]'.
    """
    assistant_prefix = build_chat_prompt(tokenizer, user_prompt)
    rationale = parse_rationale(gen_text)

    forced_prefix = f"[Rationale:]\n{rationale}\n[Answer:]"
    prefix_text = assistant_prefix + forced_prefix

    lp_yes = class_logprob_nexttoken_vllm(client, tokenizer, prefix_text, YES_VARS, topk=topk)
    lp_no  = class_logprob_nexttoken_vllm(client, tokenizer, prefix_text, NO_VARS,  topk=topk)

    if lp_yes == -float('inf') and lp_no == -float('inf'):
        return 0.5, lp_yes, lp_no

    m = max(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    return p_yes, lp_yes, lp_no


# =========================
# Main prediction loop
# =========================

def get_pred(client, data, config):
    tokenizer = client.get_tokenizer()

    p_yes_ls, lp_yes_ls, lp_no_ls, resp_ls = [], [], [], []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        if config['mode'] == 'direct':
            user_prompt = get_prompt_direct(row)
            p_yes, lp_yes, lp_no = get_yes_probability_direct_vllm(
                client, tokenizer, user_prompt, topk=config.get("logprob_topk", 20)
            )
            resp = None
        else:
            user_prompt = get_prompt(row)
            chat_prompt = build_chat_prompt(tokenizer, user_prompt)
            resp = generate_response(client, [chat_prompt])[0]
            p_yes, lp_yes, lp_no = get_yes_probability_w_rationale_vllm(
                client, tokenizer, user_prompt, resp, topk=config.get("logprob_topk", 20)
            )

        # No yes/no variant appeared in the top-k next-token logprobs; fall back to p_yes.
        if lp_yes == -float('inf') and lp_no == -float('inf'):
            print(f"[warning] no yes/no token in top-{config.get('logprob_topk', 20)} "
                  f"logprobs; using p_yes={p_yes}")

        p_yes_ls.append(p_yes)
        lp_yes_ls.append(lp_yes)
        lp_no_ls.append(lp_no)
        resp_ls.append(resp)

    data['p_yes'] = p_yes_ls
    data['logp_yes'] = lp_yes_ls
    data['logp_no'] = lp_no_ls
    data['response'] = resp_ls
    return data


# =========================
# Entry point
# =========================

def main(args):
    model_name = args.model_name
    model_id = MODEL_MAPPING.get(model_name, args.model_id)

    config = {
        'model_name': model_name,
        'model_id': model_id,
        'num_gpu': args.num_gpu,
        'mode': args.mode,
        'logprob_topk': args.logprob_topk,
    }

    client = load_model(config)

    data = pd.read_json(args.input_path, convert_dates=False)

    data = get_pred(client, data, config)

    
    out_dir = f"{args.output_path}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/{os.path.basename(args.input_path)}"
    data.to_json(save_path, orient="records")

    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--mode", choices=["direct", "rationale"], default="rationale")
    parser.add_argument("--logprob_topk", type=int, default=20,
                        help="Top-K next-token logprobs to request from vLLM.")
    args = parser.parse_args()
    main(args)
