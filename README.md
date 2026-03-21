<div align="center">

# Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?

[![Paper](https://img.shields.io/badge/arXiv-xxxx.xxxx-F46565?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxx) [![Code](https://img.shields.io/badge/GitHub-Code-000000?style=flat&logo=github&logoColor=white)](https://github.com/lasgroup/SDPO) [![W&B](https://img.shields.io/badge/W%26B-Logs-06B6D4?style=flat&logo=weightsandbiases&logoColor=white)](https://wandb.ai/jonhue/SDPO?nw=mgotcx6kk7) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-FEE47D?style=flat)](https://huggingface.co/your-repo-here)

</div>

## Introduction

## Installation

### Data Preparation
```bash
bash experiments/math/prepare_dapo_data.sh
```

### Prompt Reformatting
```bash
python experiments/math/change_math_prompts.py
```

This script reformats the default instruction-style prompts to elicit `\boxed{}` answers instead of the `Answer:` format. We found that this reformatting consistently improved the model's accuracy.

<details>
<summary>See before/after example</summary>

**Before:**
```python
[{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after "Answer:".', 'role': 'user'}]
```
**After:**
```python
[{'content': 'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\nPlease reason step by step, and put your final answer within \\boxed{}.', 'role': 'user'}]
```

</details>


### Configuration
```bash
```

## Run Examples

### GRPO and SDPO Training
```bash
```
### Evaluation
```bash
```

## Citation
If you find this work helpful, please consider citing:

## Attribution
