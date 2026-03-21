import json
import random
import os
import argparse
import re
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm.envs as envs
from utils.utils import set_seed
from utils.parser import *
from utils.math_normalization import *
from utils.grader import *

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--collected_path', type=str, required=True, help="Path to collected correct examples jsonl")
    parser.add_argument('--data_name', type=str, default="math")
    parser.add_argument('--n_sample', type=int, default=10, help="Number of examples to pick")
    parser.add_argument('--n_sampling', type=int, default=1, help="Number of responses per prompt")
    parser.add_argument('--n_rounds', type=int, default=2, help="Number of chained generation rounds")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true", default=False, help="Enable thinking mode in chat template")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs_chained')
    parser.add_argument('--no_chat_template', action='store_true')
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    args.top_k = -1 if args.temperature == 0 else args.top_k
    return args


def load_collected(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_augmented_prompt(original_messages, correct_solution):
    """Inject correct solution into the user message."""
    new_messages = []
    for msg in original_messages:
        if msg['role'] == 'user':
            new_content = (
                f"{msg['content']}\n\n"
                f"Correct solution: {correct_solution}\n\n"
                f"Correctly solve the original question."
            )
            new_messages.append({'role': 'user', 'content': new_content})
        else:
            new_messages.append(msg)
    return new_messages


def get_conversation_prompt_by_messages(tokenizer, messages, enable_thinking=False):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs['enable_thinking'] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop('enable_thinking', None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def run_one_round(sampled, llm, tokenizer, sampling_params, args, round_num, solution_key_fn):
    """
    Run one round of augmented generation.
    
    solution_key_fn: callable(example, round_num) -> str
        Returns the solution string to use as conditioning for this round.
    """
    prompt_batch = []
    for ex in sampled:
        original_messages = ex['prompt']
        solution = solution_key_fn(ex, round_num)
        augmented_messages = build_augmented_prompt(original_messages, solution)
        ex[f'_augmented_messages_round{round_num}'] = augmented_messages

        if args.no_chat_template:
            cur_prompt = "\n".join([msg['content'] for msg in augmented_messages])
        else:
            cur_prompt = get_conversation_prompt_by_messages(tokenizer, augmented_messages, enable_thinking=args.enable_thinking)
        prompt_batch.append(cur_prompt)

    print(f"\n[Round {round_num}] Generating responses...")
    completions = llm.generate(prompt_batch, sampling_params)

    round_results = []
    for i, ex in enumerate(sampled):
        new_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
        new_answers = [extract_answer(resp, args.data_name) for resp in new_responses]
        new_correct_list = [check_is_correct(ans, ex['gold_answer']) for ans in new_answers]

        # Pick the best response for next round: prefer correct ones, else use first
        best_response = None
        for j, correct in enumerate(new_correct_list):
            if correct:
                best_response = new_responses[j]
                break
        if best_response is None:
            best_response = new_responses[0]

        entry = {
            "new_responses": new_responses,
            "new_extracted_answers": new_answers,
            "new_is_correct": new_correct_list,
            "new_correct_count": sum(new_correct_list),
            "best_response": best_response,
        }
        round_results.append(entry)

    return round_results


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load collected data
    collected = load_collected(args.collected_path)
    print(f"Loaded {len(collected)} collected examples from {args.collected_path}")

    # Sample
    if len(collected) > args.n_sample:
        sampled = random.sample(collected, args.n_sample)
    else:
        sampled = collected
    print(f"Sampled {len(sampled)} examples")

    # Load model
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_sampling,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Store all round results per example
    all_rounds = {i: {} for i in range(len(sampled))}

    for round_num in range(1, args.n_rounds + 1):
        def get_solution(ex, rnd, idx=None):
            """Get the solution to condition on for this round."""
            i = sampled.index(ex)
            if rnd == 1:
                # Round 1: use original correct solution
                return ex['correct_solutions'][0]
            else:
                # Round 2+: use best response from previous round
                prev_round = all_rounds[i].get(f'round{rnd - 1}')
                if prev_round:
                    return prev_round['best_response']
                else:
                    return ex['correct_solutions'][0]

        # We need index-aware solution lookup
        prompt_batch = []
        for idx, ex in enumerate(sampled):
            original_messages = ex['prompt']
            if round_num == 1:
                solution = ex['correct_solutions'][0]
            else:
                prev = all_rounds[idx].get(f'round{round_num - 1}')
                solution = prev['best_response'] if prev else ex['correct_solutions'][0]

            augmented_messages = build_augmented_prompt(original_messages, solution)
            ex[f'_conditioning_solution_round{round_num}'] = solution

            if args.no_chat_template:
                cur_prompt = "\n".join([msg['content'] for msg in augmented_messages])
            else:
                cur_prompt = get_conversation_prompt_by_messages(tokenizer, augmented_messages, enable_thinking=args.enable_thinking)
            prompt_batch.append(cur_prompt)

        print(f"\n{'=' * 80}")
        print(f"[Round {round_num}/{args.n_rounds}] Generating responses...")
        print(f"{'=' * 80}")
        completions = llm.generate(prompt_batch, sampling_params)

        for idx, ex in enumerate(sampled):
            new_responses = [completions[idx].outputs[j].text for j in range(len(completions[idx].outputs))]
            new_answers = [extract_answer(resp, args.data_name) for resp in new_responses]
            new_correct_list = [check_is_correct(ans, ex['gold_answer']) for ans in new_answers]

            # Pick best response for next round: prefer correct, else first
            best_response = new_responses[0]
            for j, correct in enumerate(new_correct_list):
                if correct:
                    best_response = new_responses[j]
                    break

            all_rounds[idx][f'round{round_num}'] = {
                "conditioning_solution": ex.get(f'_conditioning_solution_round{round_num}', ''),
                "new_responses": new_responses,
                "new_extracted_answers": new_answers,
                "new_is_correct": new_correct_list,
                "new_correct_count": sum(new_correct_list),
                "best_response": best_response,
            }

    # Build final results
    results = []
    for idx, ex in enumerate(sampled):
        entry = {
            "question": ex['question'],
            "gold_answer": ex['gold_answer'],
            "original_correct_count": ex['correct_count'],
            "data_source": ex.get('data_source', ''),
            "original_index": ex.get('original_index', -1),
        }
        for rnd in range(1, args.n_rounds + 1):
            rkey = f'round{rnd}'
            rd = all_rounds[idx][rkey]
            entry[f'{rkey}_conditioning_solution'] = rd['conditioning_solution']  # full length
            entry[f'{rkey}_responses'] = rd['new_responses']
            entry[f'{rkey}_extracted_answers'] = rd['new_extracted_answers']
            entry[f'{rkey}_is_correct'] = rd['new_is_correct']
            entry[f'{rkey}_correct_count'] = rd['new_correct_count']
        results.append(entry)

    # Print comparison across rounds
    print(f"\n{'=' * 80}")
    print(f"CHAINED COMPARISON RESULTS ({args.n_rounds} rounds)")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        print(f"\n{'─' * 80}")
        print(f"[{i+1}/{len(results)}] Index: {r['original_index']} | Gold: {r['gold_answer']}")
        print(f"  Original: {r['original_correct_count']}/8 correct")
        for rnd in range(1, args.n_rounds + 1):
            cc = r[f'round{rnd}_correct_count']
            print(f"  Round {rnd}: {cc}/{args.n_sampling} correct")
            for j, (ans, correct) in enumerate(zip(r[f'round{rnd}_extracted_answers'], r[f'round{rnd}_is_correct'])):
                mark = "✓" if correct else "✗"
                print(f"    Response #{j+1} [{mark}] → {ans}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"  Original avg correct: {sum(r['original_correct_count'] for r in results) / len(results):.1f}/8")
    for rnd in range(1, args.n_rounds + 1):
        avg = sum(r[f'round{rnd}_correct_count'] for r in results) / len(results)
        all_correct = sum(1 for r in results if r[f'round{rnd}_correct_count'] == args.n_sampling)
        print(f"  Round {rnd} avg correct: {avg:.1f}/{args.n_sampling} | All-correct: {all_correct}/{len(results)}")
    print(f"{'=' * 80}")

    # Save per-round files
    os.makedirs(args.output_dir, exist_ok=True)

    # Save combined results
    combined_file = os.path.join(args.output_dir, "chained_results_all.jsonl")
    with open(combined_file, 'w', encoding='utf-8') as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    print(f"Saved combined: {combined_file}")

    # Save per-round files separately
    for rnd in range(1, args.n_rounds + 1):
        round_file = os.path.join(args.output_dir, f"round{rnd}_results.jsonl")
        with open(round_file, 'w', encoding='utf-8') as f:
            for idx, r in enumerate(results):
                round_entry = {
                    "question": r['question'],
                    "gold_answer": r['gold_answer'],
                    "original_correct_count": r['original_correct_count'],
                    "data_source": r['data_source'],
                    "original_index": r['original_index'],
                    "conditioning_solution": r[f'round{rnd}_conditioning_solution'],
                    "responses": r[f'round{rnd}_responses'],
                    "extracted_answers": r[f'round{rnd}_extracted_answers'],
                    "is_correct": r[f'round{rnd}_is_correct'],
                    "correct_count": r[f'round{rnd}_correct_count'],
                }
                f.write(json.dumps(round_entry, ensure_ascii=False, cls=NumpyEncoder) + "\n")
        print(f"Saved round {rnd}: {round_file}")


if __name__ == "__main__":
    main()