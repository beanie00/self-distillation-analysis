import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import os
import argparse
import vllm.envs as envs
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl
from utils.parser import *
from utils.math_normalization import *
from utils.grader import *
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def parse_list(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=8, help="n for sampling per problem")
    parser.add_argument("--max_collect", type=int, default=100, help="Stop after collecting this many examples")
    parser.add_argument("--min_correct", type=int, default=1, help="Minimum correct count to keep (inclusive)")
    parser.add_argument("--max_correct", type=int, default=4, help="Maximum correct count to keep (inclusive)")
    parser.add_argument("--data_path", type=str, required=True, help="path to parquet file")
    parser.add_argument("--data_name", type=str, default="math", help="identify how to extract answer")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--enable_thinking", action="store_true", default=False, help="Enable thinking mode in chat template")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--batch_size", default=64, type=int, help="Number of problems to process per batch")
    args = parser.parse_args()

    args.top_p = 1 if args.temperature == 0 else args.top_p
    args.top_k = -1 if args.temperature == 0 else args.top_k
    return args


def get_conversation_prompt_by_messages(tokenizer, messages, enable_thinking=False):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs['enable_thinking'] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop('enable_thinking', None)
        return tokenizer.apply_chat_template(messages, **kwargs)

def load_parquet_data(data_path):
    df = pd.read_parquet(data_path)
    return df.to_dict(orient='records')


def parse_question_new(example):
    messages = example['prompt']
    if isinstance(messages, str):
        messages = json.loads(messages)
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    return messages[-1]['content']


def parse_ground_truth_new(example):
    reward_model = example['reward_model']
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    return reward_model['ground_truth']


def get_messages(example):
    messages = example['prompt']
    if isinstance(messages, str):
        messages = json.loads(messages)
    return messages


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"Model: {model_name_or_path}")
    print(f"Sampling n={args.n_sampling} per problem")
    print(f"Collecting problems with {args.min_correct}~{args.max_correct} correct out of {args.n_sampling}, up to {args.max_collect}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_sampling,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Load data
    examples = load_parquet_data(args.data_path)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    # Output file
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    data_name_tag = os.path.splitext(os.path.basename(args.data_path))[0]
    out_file = (
        f'{args.output_dir}/{model_name}/{args.data_name}/'
        f'{data_name_tag}_think_n{args.n_sampling}_correct{args.min_correct}-{args.max_correct}_collect{args.max_collect}.jsonl'
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Load model
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0"
    print(f"Available GPUs: {available_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )

    collected = []
    total_attempted = 0
    batch_size = args.batch_size

    for batch_start in range(0, len(examples), batch_size):
        if len(collected) >= args.max_collect:
            break

        batch_end = min(batch_start + batch_size, len(examples))
        batch_examples = examples[batch_start:batch_end]

        # Build prompts
        prompt_batch = []
        for example in batch_examples:
            messages = get_messages(example)
            if args.no_chat_template:
                cur_prompt = "\n".join([msg['content'] for msg in messages])
            else:
                cur_prompt = get_conversation_prompt_by_messages(tokenizer, messages, enable_thinking=args.enable_thinking)
            prompt_batch.append(cur_prompt)

        # Generate
        completions = llm.generate(prompt_batch, sampling_params)

        # Check each problem
        for i, example in enumerate(batch_examples):
            total_attempted += 1
            question = parse_question_new(example)
            gt_ans = parse_ground_truth_new(example)
            messages = get_messages(example)

            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]

            # Extract and check answers
            generated_answers = [extract_answer(resp, args.data_name) for resp in generated_responses]
            is_correct_list = [check_is_correct(ga, gt_ans) for ga in generated_answers]
            correct_count = sum(is_correct_list)

            # Only keep if correct_count is in [min_correct, max_correct]
            if args.min_correct <= correct_count <= args.max_correct:
                # Collect all correct solutions
                correct_indices = [idx for idx, c in enumerate(is_correct_list) if c]
                correct_solutions = [generated_responses[idx] for idx in correct_indices]

                entry = {
                    "question": question,
                    "prompt": messages,
                    "gold_answer": gt_ans,
                    "correct_count": correct_count,
                    "total_samples": args.n_sampling,
                    "correct_solutions": correct_solutions,
                    "data_source": example.get("data_source", ""),
                    "original_index": batch_start + i + args.start_idx,
                }

                extra_info = example.get('extra_info', {})
                if isinstance(extra_info, str):
                    extra_info = json.loads(extra_info)
                if extra_info:
                    entry["extra_info"] = extra_info

                collected.append(entry)
                print(f"[{len(collected)}/{args.max_collect}] idx={batch_start + i}, "
                      f"correct={correct_count}/{args.n_sampling}, answer={gt_ans}")

                if len(collected) >= args.max_collect:
                    break

        print(f"Progress: {len(collected)}/{args.max_collect} collected, {total_attempted} attempted")

    # Save results
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        for d in collected:
            f.write(json.dumps(d, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    os.rename(temp_out_file, out_file)

    print("=" * 60)
    print(f"Done! Collected {len(collected)} examples out of {total_attempted} attempted")
    print(f"Filter: {args.min_correct}~{args.max_correct} correct out of {args.n_sampling}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)