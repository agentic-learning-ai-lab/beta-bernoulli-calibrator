"""
Generate verbalized probability forecasts from an input LLM.

Modes (paper defaults shown in parentheses):
  --mode default    greedy point forecast                    (T=0,   n_samples=1)   → writes 'response'
  --mode ensemble   sample n forecasts at temperature T      (T=1.0, n_samples=10)  → writes 'response_1' .. 'response_n'
  --mode w_conf     greedy forecast + verbalized confidence  (T=0,   n_samples=1)   → writes 'response'
                    (prompt also asks for a [Confidence:] line)

Examples (run from the project root):

  # Greedy initial forecast (input to BBC + the "Verbalized" baseline).
  python verbalized/generate.py \
    --model_name qwen3-8b \
    --input_path ./data/default/test.json \
    --output_path ./results/default/verbalized

  # Ensemble baseline (paper default: n=10, T=1.0).
  python verbalized/generate.py --mode ensemble \
    --model_name qwen3-8b \
    --input_path ./data/default/test.json \
    --output_path ./results/default/ensemble

  # Verbalized-confidence baseline (input to analysis/uncertainty_plot.py --mode verbalized).
  python verbalized/generate.py --mode w_conf \
    --model_name qwen3-8b \
    --input_path ./data/default/test.json \
    --output_path ./results/default/verbalized_w_conf

Set HF_HOME to control where vLLM caches model weights.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import anthropic
from vllm import LLM
from vllm.sampling_params import SamplingParams


MODEL_MAPPING = {
    'claude-4-sonnet': 'claude-sonnet-4-20250514',
    'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'qwen2_5-7b':  'Qwen/Qwen2.5-7B-Instruct',
    'qwen2_5-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen3-8b':    'Qwen/Qwen3-8B',
    'qwen3-32b':   'Qwen/Qwen3-32B',
    'future-as-label-32b': 'LightningRodLabs/future-as-label-paper-step160',
    'openforecaster-8b':   'nikhilchandak/OpenForecaster-8B',
}

LARGE_MODELS = {'llama3-70b', 'qwen2_5-72b'}

# Paper-default (temperature, n_samples) per mode.
MODE_DEFAULTS = {
    'default':  (0.0, 1),
    'ensemble': (1.0, 10),
    'w_conf':   (0.0, 1),
}

# Optional vLLM cache override. Set to a directory to download / load weights
# from <LOCAL_MODEL_DIR>/<model_id>; leave as None to let vLLM use HF_HOME.
LOCAL_MODEL_DIR = None


def load_model(config):
    model_name = config['model_name']
    model_id = config['model_id']
    if 'gpt' in model_name and 'oss' not in model_name:
        return OpenAI()
    elif 'claude' in model_name:
        return anthropic.Anthropic()
    else:
        kwargs = dict(model=model_id, tensor_parallel_size=config['num_gpu'])
        if LOCAL_MODEL_DIR:
            kwargs['download_dir'] = f'{LOCAL_MODEL_DIR}/{model_id}'
        if model_name in LARGE_MODELS:
            kwargs.update(dtype='float16', max_model_len=4096)
        elif 'gemma' in model_name:
            kwargs.update(dtype='bfloat16')
        else:
            kwargs.update(dtype='float16')
        return LLM(**kwargs)


def get_prompt(model_name, row, mode):
    base = (
        "You're an expert in forecasting events. Make a prediction of the probability that the question "
        "will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL "
        "CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number "
        "between 0 and 1.\n\n"
        "To support your reasoning, recall relevant recent events, facts, or widely known information. "
        "Ensure your rationale is well-grounded and coherent.\n\n"
        "Once you have completed your reasoning, output your answer as a number between 0 and 1."
    )

    confidence_instruction = ""
    confidence_format_line = ""
    if mode == 'w_conf':
        confidence_instruction = (
            "\n\nAfter you give your probability, also report how confident you are in that probability "
            "on a scale from 0 to 1 (0 = no confidence, 1 = extremely confident)."
        )
        confidence_format_line = "\n[Confidence:] a number between 0 and 1"

    prompt = (
        f"{base}{confidence_instruction}\n\n"
        f"Question: {row['question']}\n"
        f"Resolution Criteria: {row['resolution_criteria']}\n\n"
        f"Today's date: {row['open_date']}\n"
        f"Question close date: {row['close_date']}\n\n"
        f"Please follow the output format:\n"
        f"[Rationale:] xxx\n"
        f"[Answer:] a number between 0 and 1{confidence_format_line}"
    )

    if 'gemma' in model_name:
        return [{"role": "user", "content": prompt}]
    elif 'claude' in model_name:
        return prompt
    else:
        return [{"role": "system", "content": ""},
                {"role": "user", "content": prompt}]


def generate_response(config, client, prompt_ls, max_tokens=4096):
    """Returns List[List[str|None]] of shape [len(prompt_ls)][n_samples]."""
    model_name = config['model_name']
    model_id = config['model_id']
    n_samples = config['n_samples']
    temperature = config['temperature']
    all_responses = []

    try:
        if 'gpt' in model_name and 'oss' not in model_name:
            for prompt in tqdm(prompt_ls):
                try:
                    resp = client.chat.completions.create(
                        model=model_id, messages=prompt,
                        max_tokens=max_tokens, temperature=temperature,
                        n=n_samples,
                    )
                    samples = [c.message.content for c in resp.choices]
                    samples = (samples + [None] * n_samples)[:n_samples]
                    all_responses.append(samples)
                except Exception as e:
                    print(f"- Failed prompt: {prompt}\n- Error: {e}")
                    all_responses.append([None] * n_samples)
            return all_responses

        elif 'claude' in model_name:
            # Anthropic SDK has no `n=`; loop manually when sampling >1.
            for prompt in tqdm(prompt_ls):
                samples = []
                for _ in range(n_samples):
                    try:
                        message = client.messages.create(
                            model=model_id, max_tokens=max_tokens,
                            temperature=temperature, system='',
                            messages=[{"role": "user",
                                       "content": [{"type": "text", "text": prompt}]}],
                        )
                        samples.append(message.content[0].text)
                    except Exception as e:
                        print(f"- Failed prompt: {prompt}\n- Error: {e}")
                        samples.append(None)
                all_responses.append(samples)
            return all_responses

        else:
            sp = SamplingParams(max_tokens=max_tokens,
                                temperature=temperature, n=n_samples)
            resp = client.chat(messages=prompt_ls, sampling_params=sp, use_tqdm=True)
            for output in resp:
                samples = [output.outputs[k].text
                           for k in range(min(n_samples, len(output.outputs)))]
                samples = (samples + [None] * n_samples)[:n_samples]
                all_responses.append(samples)
            return all_responses

    except Exception as e:
        print(f"- Error: {e}")
        while len(all_responses) < len(prompt_ls):
            all_responses.append([None] * n_samples)
        return all_responses


def get_pred(client, data, config):
    prompt_ls = [get_prompt(config['model_name'], row, config['mode'])
                 for _, row in tqdm(data.iterrows(), total=len(data))]

    responses_2d = generate_response(config, client, prompt_ls)
    n_samples = config['n_samples']

    if n_samples == 1:
        data['response'] = [r[0] for r in responses_2d]
    else:
        for k in range(n_samples):
            data[f'response_{k+1}'] = [r[k] if r is not None and len(r) > k else None
                                       for r in responses_2d]
    return data


def main(args):
    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    output_path = f'{output_path}/{model_name}'
    os.makedirs(output_path, exist_ok=True)

    file_name = os.path.basename(input_path).split('.')[0]
    save_path = f'{output_path}/{file_name}_raw.json'

    model_id = MODEL_MAPPING.get(model_name, args.model_id)

    default_T, default_n = MODE_DEFAULTS[args.mode]
    temperature = args.temperature if args.temperature is not None else default_T
    n_samples = args.n_samples if args.n_samples is not None else default_n

    if args.mode != 'ensemble' and n_samples != 1:
        raise SystemExit(
            f"--n_samples > 1 is only valid with --mode ensemble "
            f"(got {n_samples} in --mode {args.mode})."
        )

    config = {
        'model_name': model_name,
        'model_id': model_id,
        'input_path': input_path,
        'save_path': save_path,
        'num_gpu': args.num_gpu,
        'mode': args.mode,
        'temperature': temperature,
        'n_samples': n_samples,
    }
    print(f"Config: {config}")

    client = load_model(config)
    data = pd.read_json(input_path, convert_dates=False)

    data = get_pred(client, data, config)
    data.to_json(save_path, orient="records")

    print(f"Evaluation results saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate verbalized forecasts from an input LLM "
                    "(default / ensemble / verbalized-confidence modes).")
    parser.add_argument('--mode', choices=['default', 'ensemble', 'w_conf'],
                        default='default',
                        help='default = single greedy; ensemble = n samples at T; '
                             'w_conf = greedy + verbalized [Confidence:].')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=False,
                        help='HF / API id, used if --model_name is not in MODEL_MAPPING.')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_gpu', type=int, default=1)

    # Mode-dependent (per-mode defaults applied in main()):
    parser.add_argument('--temperature', type=float, default=None,
                        help='Default: 0 (default/w_conf) or 1.0 (ensemble).')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Default: 1 (default/w_conf) or 10 (ensemble). '
                             'Must be 1 unless --mode ensemble.')
    args = parser.parse_args()
    main(args)
