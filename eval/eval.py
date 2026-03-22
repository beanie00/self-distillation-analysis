import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl
from utils.parser import *
from utils.math_normalization import *
from utils.grader import *
import pickle
import pandas as pd
from math import comb


def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_path", type=str, required=True, help="path to parquet file")
    parser.add_argument("--data_name", type=str, default="math", help="identify how to extract answer")
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=32768, type=int)
    # surround_with_messages: new data is already in messages format, so default is True
    parser.add_argument("--surround_with_messages", action="store_true", default=True,
                        help="Apply chat template to the prompt messages (default: True for new format)")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="If set, use raw prompt text without applying chat template")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking/reasoning mode in chat template (for QwQ, DeepSeek-R1, etc.)")
    parser.add_argument("--thinking_budget", type=int, default=None,
                        help="Max tokens for thinking part (if model supports it)")
    parser.add_argument("--output_dir", default="./outputs_off_policy", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    args = parser.parse_args()

    args.top_p = 1 if args.temperature == 0 else args.top_p
    args.top_k = -1 if args.temperature == 0 else args.top_k
    print(f"current stop list: {args.stop}")
    return args


def get_conversation_prompt_by_messages(tokenizer, messages, enable_thinking=False, thinking_budget=None):
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    # Thinking mode support (QwQ, DeepSeek-R1, etc.)
    if enable_thinking:
        kwargs['enable_thinking'] = True
        if thinking_budget is not None:
            kwargs['thinking_budget'] = thinking_budget
    else:
        # Explicitly disable thinking (if the model supports it)
        try:
            kwargs['enable_thinking'] = False
        except Exception:
            pass

    text = tokenizer.apply_chat_template(
        messages,
        **kwargs
    )
    return text


def load_parquet_data(data_path):
    """Load data from parquet file and return list of dicts."""
    df = pd.read_parquet(data_path)
    examples = df.to_dict(orient='records')
    return examples


def parse_question_new(example):
    """Extract the question text from the new prompt format.
    
    prompt is a list of message dicts, e.g.:
    [{'content': 'Solve the following math problem...', 'role': 'user'}]
    
    We return the user message content as the question.
    """
    messages = example['prompt']
    # Parse as JSON if messages is a string
    if isinstance(messages, str):
        messages = json.loads(messages)
    
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    # Fallback: return the content of the last message
    return messages[-1]['content']


def parse_ground_truth_new(example):
    """Extract ground truth answer from reward_model field.
    
    reward_model is a dict like:
    {'ground_truth': '540', 'style': 'rule-lighteval', ...}
    """
    reward_model = example['reward_model']
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    
    gt_ans = reward_model['ground_truth']
    return gt_ans


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")

    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=factor,
        top_p=args.top_p,
        
    )

    # ===== Load new data =====
    examples = load_parquet_data(args.data_path)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    data_name_tag = os.path.splitext(os.path.basename(args.data_path))[0]
    thinking_tag = "_think" if args.enable_thinking else "_nothink"
    out_file_prefix = f'{data_name_tag}_t{args.temperature}{thinking_tag}'
    out_file = f'avg_outputs/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'

    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        return

    os.makedirs(f'avg_outputs/{model_name}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)

    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # ===== Build prompts =====
    prompt_batch = []
    for example in tqdm(examples, total=len(examples), desc="Building prompts"):
        messages = example['prompt']
        if isinstance(messages, str):
            messages = json.loads(messages)

        if args.no_chat_template:
            # Use raw content without chat template
            cur_prompt = "\n".join([msg['content'] for msg in messages])
        else:
            # Apply chat template (default)
            cur_prompt = get_conversation_prompt_by_messages(
                tokenizer=tokenizer,
                messages=messages,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
            )

        prompt_batch.append(cur_prompt)

    print("=" * 50)
    print("Example prompt:")
    print(prompt_batch[0][:500])
    print("=" * 50)

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )

    file_outputs = []
    correct_cnt = 0
    total_tokens = 0
    total_responses = 0

    for cur_generation_epoch in range(generation_epoch):
        completions_save_file = (
            f'{args.completions_save_dir}/{model_name}/{args.data_name}/'
            f'{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}_gen_round{cur_generation_epoch}.pkl'
        )

        completions = llm.generate(prompt_batch, sampling_params)
        save_completions(completions, completions_save_file)

        for i in range(len(examples)):
            question = parse_question_new(examples[i])
            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]

            # If thinking mode is enabled, separate thinking and answer parts
            thinking_contents = []
            answer_responses = []
            if args.enable_thinking:
                for response in generated_responses:
                    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                    if think_match:
                        thinking_contents.append(think_match.group(1).strip())
                        # Extract the part after the think tags as the answer
                        answer_part = response[think_match.end():].strip()
                        answer_responses.append(answer_part)
                    else:
                        thinking_contents.append("")
                        answer_responses.append(response)
            else:
                answer_responses = generated_responses

            # Calculate token lengths
            response_token_lengths = []
            thinking_token_lengths = []
            for response in generated_responses:
                tokens = tokenizer.encode(response, add_special_tokens=False)
                response_token_lengths.append(len(tokens))
                total_tokens += len(tokens)
                total_responses += 1
            
            if args.enable_thinking:
                for think_content in thinking_contents:
                    if think_content:
                        think_tokens = tokenizer.encode(think_content, add_special_tokens=False)
                        thinking_token_lengths.append(len(think_tokens))
                    else:
                        thinking_token_lengths.append(0)

            if cur_generation_epoch == 0:
                output_entry = {
                    "question": question,
                    "generated_responses": generated_responses,
                    "answer_responses": answer_responses,  # Answer part with thinking removed
                    "response_token_lengths": response_token_lengths,
                }
                if args.enable_thinking:
                    output_entry["thinking_contents"] = thinking_contents
                    output_entry["thinking_token_lengths"] = thinking_token_lengths
                
                file_outputs.append(output_entry)
                # Store additional info (id, source, etc.) from extra_info
                extra_info = examples[i].get('extra_info', {})
                if isinstance(extra_info, str):
                    extra_info = json.loads(extra_info)
                if extra_info:
                    file_outputs[i]["extra_info"] = extra_info
                file_outputs[i]["data_source"] = examples[i].get("data_source", "")
            else:
                file_outputs[i]['generated_responses'] += generated_responses
                file_outputs[i]['answer_responses'] += answer_responses
                file_outputs[i]['response_token_lengths'] += response_token_lengths
                if args.enable_thinking:
                    file_outputs[i]['thinking_contents'] += thinking_contents
                    file_outputs[i]['thinking_token_lengths'] += thinking_token_lengths

    print("llm generate done")
    print(f"Total examples: {len(file_outputs)}")

    avg_acc_list = []
    k = args.k

    for i in tqdm(range(len(examples)), "check correct..."):
        gt_ans = parse_ground_truth_new(examples[i])
        # Use answer_responses in thinking mode, otherwise use generated_responses
        responses_for_extraction = file_outputs[i]['answer_responses']
        generated_answers = [extract_answer(resp, args.data_name) for resp in responses_for_extraction]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

        # Calculate average token length per problem
        avg_token_length = sum(file_outputs[i]['response_token_lengths']) / len(file_outputs[i]['response_token_lengths'])
        file_outputs[i]['avg_response_token_length'] = avg_token_length
        
        if args.enable_thinking and 'thinking_token_lengths' in file_outputs[i]:
            avg_thinking_length = sum(file_outputs[i]['thinking_token_lengths']) / len(file_outputs[i]['thinking_token_lengths'])
            file_outputs[i]['avg_thinking_token_length'] = avg_thinking_length

        if len(is_correct_list) > 1:
            avg_acc = sum(is_correct_list) / len(is_correct_list)
            avg_acc_list.append(avg_acc)
            file_outputs[i]['avg_accuracy'] = avg_acc

    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)

    # Calculate overall average token length
    avg_response_length = total_tokens / total_responses if total_responses > 0 else 0

    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Pass@{args.n_sampling}: {correct_cnt / len(examples):.4f}")
    print(f"Average Response Length (tokens): {avg_response_length:.2f}")
    print(f"Thinking Mode: {'ON' if args.enable_thinking else 'OFF'}")

    if args.enable_thinking:
        all_thinking_lengths = []
        for fo in file_outputs:
            if 'thinking_token_lengths' in fo:
                all_thinking_lengths.extend(fo['thinking_token_lengths'])
        if all_thinking_lengths:
            avg_thinking = sum(all_thinking_lengths) / len(all_thinking_lengths)
            print(f"Average Thinking Length (tokens): {avg_thinking:.2f}")

    if avg_acc_list:
        overall_avg = sum(avg_acc_list) / len(avg_acc_list)
        print(f"Avg@{args.n_sampling}: {overall_avg:.4f}")
    else:
        print(f"Pass@1: {correct_cnt / len(examples):.4f}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)