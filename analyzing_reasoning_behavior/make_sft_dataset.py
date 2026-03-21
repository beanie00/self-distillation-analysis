import json
import argparse
import os


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_system_prompt(messages):
    """Extract system prompt from original messages if exists, else use default."""
    for msg in messages:
        if msg['role'] == 'system':
            return msg['content']
    return "Please reason step by step, and put your final answer within \\boxed{}."


def extract_instruction(messages):
    """Extract user instruction (question) from messages."""
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    return messages[-1]['content']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to comparison_results.jsonl")
    parser.add_argument('--output_dir', type=str, default='./sft_datasets')
    parser.add_argument('--prev_name', type=str, default='prev_correct.json')
    parser.add_argument('--new_name', type=str, default='new_correct.json')
    args = parser.parse_args()

    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} examples from {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)

    prev_dataset = []
    new_dataset = []

    for r in data:
        messages = r.get('prompt', [])
        instruction = extract_instruction(messages) if messages else r['question']
        system = extract_system_prompt(messages) if messages else "Please reason step by step, and put your final answer within \\boxed{}."

        # Previous correct solution
        prev_entry = {
            "instruction": instruction,
            "input": "",
            "output": r['round1_conditioning_solution'],
            "system": system,
        }
        prev_dataset.append(prev_entry)

        # Pick the first correct response from round 1, skip if none
        correct_resp = None
        for resp, is_correct in zip(r['round1_responses'], r['round1_is_correct']):
            if is_correct:
                correct_resp = resp
                break # skip problems with no correct response in round 1

        if correct_resp is None:
            continue 

        new_entry = {
            "instruction": instruction,
            "input": "",
            "output": correct_resp,
            "system": system,
        }
        new_dataset.append(new_entry)

    # Save
    prev_path = os.path.join(args.output_dir, args.prev_name)
    new_path = os.path.join(args.output_dir, args.new_name)

    with open(prev_path, 'w', encoding='utf-8') as f:
        json.dump(prev_dataset, f, ensure_ascii=False, indent=2)

    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    # Count how many examples have at least one correct response in round 1
    new_correct_cnt = sum(1 for r in data if any(r['round1_is_correct']))
    print(f"prev_correct: {len(prev_dataset)} examples → {prev_path}")
    print(f"new_responses: {len(new_dataset)} examples → {new_path}  (with correct round1: {new_correct_cnt}, without: {len(data) - new_correct_cnt})")


if __name__ == "__main__":
    main()