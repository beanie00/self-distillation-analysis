import json
import argparse
import os
from transformers import AutoTokenizer


THINKING_WORDS = [
    'wait', 'hmm', 'perhaps', 'maybe', 'actually',
    'alternatively', 'seems', 'might', 'likely', 'check'
]


def count_thinking_words(text):
    """Count thinking/hedging words (case-insensitive)."""
    text_lower = text.lower()
    counts = {}
    total = 0
    for word in THINKING_WORDS:
        c = text_lower.count(word)
        counts[word] = c
        total += c
    return counts, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./outputs_chained/chained_results_answer_only.jsonl')
    parser.add_argument('--output', type=str, default='./outputs_chained/chained_report2.txt')
    parser.add_argument('--model_name_or_path', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument('--n_rounds', type=int, default=1, help="Number of rounds in the chained results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Detect n_rounds from data if not specified
    n_rounds = args.n_rounds
    # Verify by checking keys
    for rnd in range(1, n_rounds + 1):
        key = f'round{rnd}_responses'
        if key not in data[0]:
            print(f"Warning: '{key}' not found in data. Adjusting n_rounds to {rnd - 1}")
            n_rounds = rnd - 1
            break

    print(f"Analyzing {len(data)} problems across {n_rounds} rounds")

    # Pre-compute stats per problem per round
    problem_stats = []
    for r in data:
        ps = {}
        for rnd in range(1, n_rounds + 1):
            rkey = f'round{rnd}'
            responses = r[f'{rkey}_responses']
            round_stats = []
            for resp in responses:
                ntok = count_tokens(resp)
                wc, wt = count_thinking_words(resp)
                round_stats.append({'tokens': ntok, 'word_counts': wc, 'word_total': wt})

            avg_tokens = sum(s['tokens'] for s in round_stats) / len(round_stats)
            avg_word_total = sum(s['word_total'] for s in round_stats) / len(round_stats)

            ps[rkey] = {
                'stats': round_stats,
                'avg_tokens': avg_tokens,
                'avg_word_total': avg_word_total,
                'correct_count': r[f'{rkey}_correct_count'],
                'n_responses': len(responses),
            }
        problem_stats.append(ps)

    with open(args.output, 'w', encoding='utf-8') as out:
        # ===== HEADER =====
        out.write("=" * 140 + "\n")
        out.write(f"  CHAINED COMPARISON REPORT: {n_rounds}-Round Generation\n")
        out.write("=" * 140 + "\n\n")

        # ===== SUMMARY TABLE =====
        out.write("SUMMARY TABLE\n")
        out.write("─" * 140 + "\n")

        # Build header
        header = f"{'#':<4} {'Idx':<7} {'Gold':<18} {'Orig':<7}"
        for rnd in range(1, n_rounds + 1):
            header += f" {'R'+str(rnd)+'Cor':<7} {'R'+str(rnd)+'Tok':<9}"
        header += f" {'TokΔ(1→'+str(n_rounds)+')':<12} {'Result':<10}"
        out.write(header + "\n")
        out.write("─" * 140 + "\n")

        for i, (r, ps) in enumerate(zip(data, problem_stats)):
            orig = r['original_correct_count']
            gold = str(r['gold_answer'])[:16]

            # Determine result based on last round vs first round
            last_correct = ps[f'round{n_rounds}']['correct_count']
            last_total = ps[f'round{n_rounds}']['n_responses']
            first_correct = ps['round1']['correct_count']

            if last_correct == last_total:
                change = "★ ALL"
            elif last_correct > first_correct:
                change = "▲ UP"
            elif last_correct < first_correct:
                change = "▼ DOWN"
            elif last_correct == 0:
                change = "✗ NONE"
            else:
                change = "─"

            row = f"{i+1:<4} {r.get('original_index', '?'):<7} {gold:<18} {orig}/8{'':<3}"
            for rnd in range(1, n_rounds + 1):
                rkey = f'round{rnd}'
                cc = ps[rkey]['correct_count']
                nr = ps[rkey]['n_responses']
                at = ps[rkey]['avg_tokens']
                row += f" {cc}/{nr}{'':<3} {at:<9.0f}"

            tok_diff = ps[f'round{n_rounds}']['avg_tokens'] - ps['round1']['avg_tokens']
            row += f" {tok_diff:<+12.0f} {change}"
            out.write(row + "\n")

        # ===== OVERALL STATS =====
        out.write("─" * 140 + "\n\n")
        out.write("OVERALL STATISTICS\n")
        out.write("─" * 140 + "\n")

        total_orig = sum(r['original_correct_count'] for r in data)
        out.write(f"  Original avg correct: {total_orig/len(data):.1f}/8\n\n")

        for rnd in range(1, n_rounds + 1):
            rkey = f'round{rnd}'
            total_correct = sum(ps[rkey]['correct_count'] for ps in problem_stats)
            total_responses = sum(ps[rkey]['n_responses'] for ps in problem_stats)
            avg_tok = sum(ps[rkey]['avg_tokens'] for ps in problem_stats) / len(problem_stats)
            avg_words = sum(ps[rkey]['avg_word_total'] for ps in problem_stats) / len(problem_stats)
            all_correct = sum(1 for ps in problem_stats if ps[rkey]['correct_count'] == ps[rkey]['n_responses'])
            none_correct = sum(1 for ps in problem_stats if ps[rkey]['correct_count'] == 0)

            out.write(f"  Round {rnd}:\n")
            out.write(f"    Correct: {total_correct}/{total_responses}\n")
            out.write(f"    All correct: {all_correct}/{len(data)} | None correct: {none_correct}/{len(data)}\n")
            out.write(f"    Avg tokens: {avg_tok:.0f} | Avg thinking words: {avg_words:.1f}\n\n")

        # Round-over-round comparison
        if n_rounds >= 2:
            out.write("  Round-over-round changes:\n")
            for rnd in range(2, n_rounds + 1):
                prev_rkey = f'round{rnd-1}'
                curr_rkey = f'round{rnd}'
                tok_diff = (
                    sum(ps[curr_rkey]['avg_tokens'] for ps in problem_stats) -
                    sum(ps[prev_rkey]['avg_tokens'] for ps in problem_stats)
                ) / len(problem_stats)
                word_diff = (
                    sum(ps[curr_rkey]['avg_word_total'] for ps in problem_stats) -
                    sum(ps[prev_rkey]['avg_word_total'] for ps in problem_stats)
                ) / len(problem_stats)
                correct_diff = (
                    sum(ps[curr_rkey]['correct_count'] for ps in problem_stats) -
                    sum(ps[prev_rkey]['correct_count'] for ps in problem_stats)
                )
                improved = sum(
                    1 for ps in problem_stats
                    if ps[curr_rkey]['correct_count'] > ps[prev_rkey]['correct_count']
                )
                degraded = sum(
                    1 for ps in problem_stats
                    if ps[curr_rkey]['correct_count'] < ps[prev_rkey]['correct_count']
                )
                out.write(
                    f"    Round {rnd-1} → {rnd}: "
                    f"tokens {tok_diff:+.0f} | thinking words {word_diff:+.1f} | "
                    f"correct {correct_diff:+d} | "
                    f"improved {improved} | degraded {degraded}\n"
                )
        out.write("\n\n")

        # ===== THINKING WORD BREAKDOWN PER ROUND =====
        out.write("=" * 140 + "\n")
        out.write("  THINKING WORD BREAKDOWN (aggregated across all problems)\n")
        out.write("=" * 140 + "\n\n")

        # Header
        word_header = f"{'Word':<15}"
        for rnd in range(1, n_rounds + 1):
            word_header += f" {'R'+str(rnd)+' Total':<12} {'R'+str(rnd)+' Avg':<12}"
        word_header += f" {'Trend(1→'+str(n_rounds)+')':<12}"
        out.write(word_header + "\n")
        out.write("─" * 140 + "\n")

        for word in THINKING_WORDS:
            row = f"{word:<15}"
            round_avgs = []
            for rnd in range(1, n_rounds + 1):
                rkey = f'round{rnd}'
                total_sum = sum(
                    sum(ns['word_counts'][word] for ns in ps[rkey]['stats'])
                    for ps in problem_stats
                )
                # Per-problem average (averaging over n_sampling per problem)
                per_problem_avg = sum(
                    sum(ns['word_counts'][word] for ns in ps[rkey]['stats']) / len(ps[rkey]['stats'])
                    for ps in problem_stats
                ) / len(problem_stats)
                round_avgs.append(per_problem_avg)
                row += f" {total_sum:<12} {per_problem_avg:<12.1f}"

            # Trend: compare last round vs first round
            first_avg = round_avgs[0]
            last_avg = round_avgs[-1]
            if first_avg > 0:
                ratio = last_avg / first_avg
                if ratio > 1.3:
                    trend = "📈 증가"
                elif ratio < 0.7:
                    trend = "📉 감소"
                else:
                    trend = "↔ 유사"
            else:
                trend = "NEW" if last_avg > 0 else "─"
            row += f" {trend}"
            out.write(row + "\n")

        # Totals row
        out.write("─" * 140 + "\n")
        totals_row = f"{'TOTAL':<15}"
        for rnd in range(1, n_rounds + 1):
            rkey = f'round{rnd}'
            total_all = sum(
                sum(ns['word_total'] for ns in ps[rkey]['stats'])
                for ps in problem_stats
            )
            totals_row += f" {total_all:<12} {'':<12}"
        out.write(totals_row + "\n")
        out.write("\n\n")

        # ===== PER-PROBLEM ROUND TRAJECTORY =====
        out.write("=" * 140 + "\n")
        out.write("  PER-PROBLEM TRAJECTORY\n")
        out.write("=" * 140 + "\n\n")

        for i, (r, ps) in enumerate(zip(data, problem_stats)):
            out.write(f"[{i+1}] Index: {r.get('original_index', '?')} | Gold: {r['gold_answer']} | Original: {r['original_correct_count']}/8\n")
            for rnd in range(1, n_rounds + 1):
                rkey = f'round{rnd}'
                cc = ps[rkey]['correct_count']
                nr = ps[rkey]['n_responses']
                at = ps[rkey]['avg_tokens']
                aw = ps[rkey]['avg_word_total']
                answers = r[f'{rkey}_extracted_answers']
                corrects = r[f'{rkey}_is_correct']
                answer_str = " | ".join(
                    f"{'✓' if c else '✗'} {a}" for a, c in zip(answers, corrects)
                )
                out.write(
                    f"  Round {rnd}: {cc}/{nr} correct | "
                    f"{at:.0f} tok | {aw:.1f} think-words | "
                    f"Answers: [{answer_str}]\n"
                )
            out.write("\n")

    print(f"Report saved to: {args.output}")
    print(f"Total problems: {len(data)}, Rounds: {n_rounds}")


if __name__ == "__main__":
    main()