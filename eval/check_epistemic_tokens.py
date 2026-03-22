import json
import argparse
from collections import defaultdict

EPISTEMIC_WORDS = [
    'wait', 'hmm', 'perhaps', 'maybe', 'actually',
    'alternatively', 'seems', 'might', 'likely', 'check'
]


def count_epistemic_words(text):
    text_lower = text.lower()
    counts = {}
    total = 0
    for word in EPISTEMIC_WORDS:
        c = text_lower.count(word)
        counts[word] = c
        total += c
    return counts, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="avg_outputs/math-SDPO-Qwen3-8B-think/global_step_100/output_hf_model/math/test_t0.6_think_k16_s0_e30.jsonl")
    parser.add_argument('--response_key', type=str, default='generated_responses',
                        help='Key for the response list in JSONL (e.g., generated_responses, answer_responses)')
    args = parser.parse_args()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    global_word_counts = defaultdict(int)
    global_total = 0
    total_responses = 0

    per_problem = []

    for i, record in enumerate(data):
        responses = record.get(args.response_key, [])
        prob_counts = defaultdict(int)
        prob_total = 0

        for resp in responses:
            counts, total = count_epistemic_words(resp)
            for w, c in counts.items():
                prob_counts[w] += c
                global_word_counts[w] += c
            prob_total += total
            global_total += total
            total_responses += 1

        n = len(responses) or 1
        per_problem.append({
            'index': i,
            'num_responses': len(responses),
            'word_totals': dict(prob_counts),
            'total': prob_total,
            'avg': prob_total / n,
        })

    # === Print Report ===
    print(f"Problems: {len(data)} | Responses: {total_responses}\n")

    print(f"{'Word':<18} {'Total':<10} {'Avg/Resp':<12} {'Avg/Problem':<12}")
    print("─" * 52)
    for word in EPISTEMIC_WORDS:
        t = global_word_counts[word]
        print(f"{word:<18} {t:<10} {t/total_responses if total_responses else 0:<12.3f} {t/len(data) if data else 0:<12.3f}")
    print("─" * 52)
    print(f"{'TOTAL':<18} {global_total:<10} {global_total/total_responses if total_responses else 0:<12.2f} {global_total/len(data) if data else 0:<12.2f}")


if __name__ == "__main__":
    main()